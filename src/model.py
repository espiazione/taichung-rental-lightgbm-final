from __future__ import annotations

import time
import datetime
from functools import lru_cache
from typing import Any

import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import (
    ARTIFACT_DIR,
    DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    MODEL_ARTIFACT_PATH,
    SEED,
    TARGET_COLUMN,
    USER_CONTINUOUS,
    USER_BINARY,
    SPATIAL_VARS,
    POI_LAYERS,
)

# ==========================================
# 🌟 專門處理中文樓層的函數
# ==========================================
def _convert_chinese_floor(text):
    if pd.isna(text): return np.nan
    text = str(text).replace('層', '').strip()

    if any(word in text for word in ['見其他', '全', '夾', '屋頂', '避難', '平台', '通道']):
        return np.nan

    is_basement = -1 if '地下' in text else 1
    text = text.replace('地下', '')
    cn_num = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}

    try:
        val = 0
        if text == '十': val = 10
        elif len(text) == 1: val = cn_num.get(text, np.nan)
        elif len(text) == 2:
            if text[0] == '十': val = 10 + cn_num.get(text[1], 0)
            elif text[1] == '十': val = cn_num.get(text[0], 0) * 10
        elif len(text) == 3 and text[1] == '十':
            val = cn_num.get(text[0], 0) * 10 + cn_num.get(text[2], 0)
        else:
            return np.nan
        return val * is_basement
    except:
        return np.nan

# ==========================================
# 🌟 快取行政區圖層
# ==========================================
@lru_cache(maxsize=1)
def _get_town_gdf():
    path, layer = POI_LAYERS["towns"]
    return gpd.read_file(path)

# ==========================================
# 模型資料準備核心
# ==========================================
def _prepare_training_frame() -> tuple[pd.DataFrame, pd.Series, dict[str, dict[str, float]]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到訓練資料：{DATA_PATH}")
    
    print("讀取空間資料庫中...")
    gdf = gpd.read_file(DATA_PATH)
    
    # --- 🌟 新增：處理時間與行政區特徵 ---
    if 'transaction year month and day' in gdf.columns:
        gdf['transaction_year'] = gdf['transaction year month and day'].astype(str).str[:3].astype(int) + 1911
    else:
        gdf['transaction_year'] = 2026

    if 'town' in gdf.columns:
        gdf['town'] = gdf['town'].astype('category')
    
    # --- 🌟 樓層特徵工程 ---
    if 'floor_ratio' not in gdf.columns:
        print("🏗️ 正在執行進階樓層特徵工程 (處理中文樓層與透天厝)...")
        shift_col = 'shifting level' if 'shifting level' in gdf.columns else ('移轉層次' if '移轉層次' in gdf.columns else None)
        total_col = 'total floor number' if 'total floor number' in gdf.columns else ('總樓層數' if '總樓層數' in gdf.columns else None)
        btype_col = 'b_type' if 'b_type' in gdf.columns else ('建物型態' if '建物型態' in gdf.columns else None)
        
        if shift_col and total_col and btype_col:
            gdf['floor_num'] = gdf[shift_col].apply(_convert_chinese_floor)
            gdf['total_floor_num'] = gdf[total_col].apply(_convert_chinese_floor)
            is_townhouse = gdf[btype_col].astype(str).str.contains('透天厝', na=False)
            gdf.loc[is_townhouse & gdf['floor_num'].isna(), 'floor_num'] = 4
            gdf.loc[is_townhouse & gdf['total_floor_num'].isna(), 'total_floor_num'] = 4
            gdf['floor_num'] = gdf['floor_num'].fillna(1)
            gdf['total_floor_num'] = gdf['total_floor_num'].fillna(1)
            gdf['floor_ratio'] = np.where(gdf['total_floor_num'] > 0, gdf['floor_num'] / gdf['total_floor_num'], 1)
            gdf['is_top_floor'] = np.where(gdf['floor_num'] == gdf['total_floor_num'], 1, 0)
        else:
            gdf['floor_ratio'] = 0.5
            
    # --- 🌟 動態展開類別變數 (Auto One-Hot Encoding) ---
    print("🔄 檢查並自動展開類別特徵 (One-Hot Encoding)...")
    if 'b_type' in gdf.columns:
        print("  -> 展開 b_type")
        dummies_b = pd.get_dummies(gdf['b_type'], prefix='b_type').astype(float)
        gdf = pd.concat([gdf, dummies_b], axis=1)
    elif '建物型態' in gdf.columns:
        print("  -> 展開 建物型態")
        dummies_b = pd.get_dummies(gdf['建物型態'], prefix='b_type').astype(float)
        gdf = pd.concat([gdf, dummies_b], axis=1)

    if 'transaction sign' in gdf.columns:
        print("  -> 展開 transaction sign")
        dummies_t = pd.get_dummies(gdf['transaction sign'], prefix='transaction sign').astype(float)
        gdf = pd.concat([gdf, dummies_t], axis=1)
    elif '交易標的' in gdf.columns:
        print("  -> 展開 交易標的")
        dummies_t = pd.get_dummies(gdf['交易標的'], prefix='transaction sign').astype(float)
        gdf = pd.concat([gdf, dummies_t], axis=1)

    # --- 🌟 自動計算對數 (ln_) ---
    missing_initial = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in gdf.columns]
    for m in missing_initial:
        if m.startswith('ln_'):
            if m == 'ln_B_area' and 'B_area_m2' in gdf.columns:
                print(f"📐 正在自動補算對數特徵: {m} (精準對應 B_area_m2)")
                gdf[m] = np.log(pd.to_numeric(gdf['B_area_m2'], errors='coerce').clip(lower=1e-5))
                gdf[m] = gdf[m].fillna(0)
            else:
                base_col = m[3:] 
                if base_col in gdf.columns:
                    print(f"📐 正在自動補算對數特徵: {m} (來自 {base_col})")
                    gdf[m] = np.log(pd.to_numeric(gdf[base_col], errors='coerce').clip(lower=1e-5))
                    gdf[m] = gdf[m].fillna(0)

    # ==========================================
    # 🧹 極端異常值清道夫
    # ==========================================
    print("🧹 執行資料清洗：過濾極端異常房價...")
    before_drop = len(gdf)
    
    if 'u_price' in gdf.columns:
        gdf = gdf[gdf['u_price'] > 1000]
        pass 
        
    if TARGET_COLUMN in gdf.columns:
        gdf = gdf[gdf[TARGET_COLUMN] > -5] 

    after_drop = len(gdf)
    print(f"   -> 移除了 {before_drop - after_drop} 筆異常幽靈資料！")

    # --- 🌟 最終容錯檢查 ---
    missing_final = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in gdf.columns]
    for m in missing_final:
        if m in SPATIAL_VARS or m in USER_BINARY:
            gdf[m] = 0.0
        else:
            raise ValueError(f"訓練資料仍缺少必要核心欄位：{m}。請檢查資料庫！")
            
    gdf = gdf.dropna(subset=[TARGET_COLUMN]).copy()
    X = gdf[FEATURE_COLUMNS].copy()
    y = gdf[TARGET_COLUMN].copy()
    
    stats: dict[str, dict[str, float]] = {}
    for col in USER_CONTINUOUS:
        if col in X.columns:
            series = pd.to_numeric(X[col], errors="coerce").fillna(0)
            q01, q99 = float(series.quantile(0.01)), float(series.quantile(0.99))
            if np.isclose(q01, q99):
                q01, q99 = float(series.min()), float(series.max())
            stats[col] = {
                "min": round(q01, 2),
                "max": round(q99, 2),
                "median": round(float(series.median()), 2),
                "step": 1.0 if series.dtype == int else 0.1,
            }
            
    return X, y, stats

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MAPE_pct": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
    }

def train_and_save_model(artifact_path=MODEL_ARTIFACT_PATH) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, stats = _prepare_training_frame()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    print("🚀 開始訓練 LightGBM 房價預測模型...")
    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    
    # 🌟 確保模型知道行政區是類別特徵
    cat_features = ['town'] if 'town' in X.columns else None
    model.fit(X_train, y_train, categorical_feature=cat_features)
    
    pred = model.predict(X_test)
    metrics = _metrics(y_test, pred)
    
    feature_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS, 
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    shap_importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Mean_Abs_SHAP": np.nan})
    try:
        import shap
        print("📊 計算 SHAP 影響力中...")
        sample_n = min(3000, len(X_test))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:sample_n], check_additivity=False)
        shap_importance = pd.DataFrame({
            "Feature": FEATURE_COLUMNS, 
            "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0)
        }).sort_values("Mean_Abs_SHAP", ascending=False)
    except Exception as exc:
        print(f"SHAP 計算跳過: {exc}")
        
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_stats": stats,
        "feature_importance": feature_importance.reset_index(drop=True),
        "shap_importance": shap_importance.reset_index(drop=True),
        "trained_rows": len(X),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # 🌟 核心修正：儲存訓練時的行政區清單
        "town_categories": list(X['town'].cat.categories) if 'town' in X.columns else []
    }
    joblib.dump(bundle, artifact_path)
    print("✅ 模型訓練完成並儲存成功！")
    return bundle

@lru_cache(maxsize=1)
def load_or_train_bundle() -> dict[str, Any]:
    if MODEL_ARTIFACT_PATH.exists():
        return joblib.load(MODEL_ARTIFACT_PATH)
    return train_and_save_model(MODEL_ARTIFACT_PATH)

def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Feature": col, "說明": FEATURE_DESCRIPTIONS.get(col, "")} for col in FEATURE_COLUMNS]
    )

def predict_rent_per_ping(
    bundle: dict[str, Any],
    user_inputs: dict[str, float],
    location_features: dict[str, float],
    x: float,
    y: float,
) -> dict[str, Any]:
    
    # 1. 空間定位：自動尋找正確欄位名稱
    town_gdf = _get_town_gdf()
    point = Point(x, y)
    match = town_gdf[town_gdf.contains(point)]
    
    town_name = "西屯區"
    if not match.empty:
        possible_cols = ['town', 'TOWNNAME', 'TOWN', 'TOWN_NAME', 'townname', '鄉鎮市區', '鄉鎮名稱', 'NAME', 'name']
        for col in possible_cols:
            if col in match.columns:
                town_name = match.iloc[0][col]
                break
    
    # 2. 組合特徵
    row_dict = {}
    row_dict.update({key: float(value) for key, value in user_inputs.items() if key != 'town' and key != 'transaction_year'})
    row_dict.update({key: float(value) for key, value in location_features.items()})
    
    row_dict['town'] = town_name
    row_dict['transaction_year'] = float(datetime.datetime.now().year)
    
    for col in FEATURE_COLUMNS:
        if col not in row_dict:
            row_dict[col] = 0.0
            
    model_row = pd.DataFrame([row_dict])[FEATURE_COLUMNS]
    
    # 🌟 核心修正：強制使用訓練時的類別清單對齊
    if 'town' in model_row.columns:
        train_cats = bundle.get("town_categories", [])
        if train_cats:
            model_row['town'] = pd.Categorical(model_row['town'], categories=train_cats)
        else:
            model_row['town'] = model_row['town'].astype('category')
            
    ln_pred = float(bundle["model"].predict(model_row)[0])
    
    return {
        "ln_u_price": ln_pred,
        "price_per_ping": float(np.exp(ln_pred)), 
        "raw_features": row_dict,
        "detected_town": town_name,
    }