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

# 🌟 確保 'town' 絕對在特徵清單中
if 'town' not in FEATURE_COLUMNS:
    FEATURE_COLUMNS.append('town')

# 🌟 全台中市行政區固定整數映射表 (純數值化，免疫 category 錯誤)
TOWN_MAPPING = {
    '西屯區': 0, '北屯區': 1, '南屯區': 2, '東區': 3, '西區': 4, '南區': 5, '北區': 6, '中區': 7,
    '豐原區': 8, '大里區': 9, '太平區': 10, '清水區': 11, '沙鹿區': 12, '大甲區': 13, '梧棲區': 14,
    '烏日區': 15, '神岡區': 16, '大雅區': 17, '潭子區': 18, '大肚區': 19, '龍井區': 20, '霧峰區': 21,
    '后里區': 22, '石岡區': 23, '東勢區': 24, '和平區': 25, '新社區': 26, '外埔區': 27, '大安區': 28
}

def _convert_chinese_floor(text):
    if pd.isna(text): return np.nan
    text = str(text).replace('層', '').strip()
    if any(word in text for word in ['見其他', '全', '夾', '屋頂', '避難', '平台', '通道']): return np.nan
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
        else: return np.nan
        return val * is_basement
    except: return np.nan

@lru_cache(maxsize=1)
def _get_town_gdf():
    path, layer = POI_LAYERS["towns"]
    return gpd.read_file(path)

def _prepare_training_frame() -> tuple[pd.DataFrame, pd.Series, dict[str, dict[str, float]]]:
    if not DATA_PATH.exists(): raise FileNotFoundError(f"找不到訓練資料：{DATA_PATH}")
    
    print("讀取空間資料庫中...")
    gdf = gpd.read_file(DATA_PATH)
    
    if 'transaction year month and day' in gdf.columns:
        gdf['transaction_year'] = gdf['transaction year month and day'].astype(str).str[:3].astype(int) + 1911
    else:
        gdf['transaction_year'] = 2026

    # 將 town 轉為數字
    if 'town' in gdf.columns:
        gdf['town'] = gdf['town'].map(TOWN_MAPPING).fillna(0).astype(float)
    else:
        gdf['town'] = 0.0
    
    if 'floor_ratio' not in gdf.columns:
        print("🏗️ 正在執行進階樓層特徵工程...")
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
            
    print("🔄 自動展開類別特徵...")
    if 'b_type' in gdf.columns:
        dummies_b = pd.get_dummies(gdf['b_type'], prefix='b_type').astype(float)
        gdf = pd.concat([gdf, dummies_b], axis=1)
    elif '建物型態' in gdf.columns:
        dummies_b = pd.get_dummies(gdf['建物型態'], prefix='b_type').astype(float)
        gdf = pd.concat([gdf, dummies_b], axis=1)

    if 'transaction sign' in gdf.columns:
        dummies_t = pd.get_dummies(gdf['transaction sign'], prefix='transaction sign').astype(float)
        gdf = pd.concat([gdf, dummies_t], axis=1)
    elif '交易標的' in gdf.columns:
        dummies_t = pd.get_dummies(gdf['交易標的'], prefix='transaction sign').astype(float)
        gdf = pd.concat([gdf, dummies_t], axis=1)

    missing_initial = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in gdf.columns]
    for m in missing_initial:
        if m.startswith('ln_'):
            if m == 'ln_B_area' and 'B_area_m2' in gdf.columns:
                gdf[m] = np.log(pd.to_numeric(gdf['B_area_m2'], errors='coerce').clip(lower=1e-5))
            else:
                base_col = m[3:] 
                if base_col in gdf.columns:
                    gdf[m] = np.log(pd.to_numeric(gdf[base_col], errors='coerce').clip(lower=1e-5))
            gdf[m] = gdf[m].fillna(0)

    print("🧹 過濾極端異常房價...")
    if 'u_price' in gdf.columns: gdf = gdf[gdf['u_price'] > 1000]
    if TARGET_COLUMN in gdf.columns: gdf = gdf[gdf[TARGET_COLUMN] > -5] 

    missing_final = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in gdf.columns]
    for m in missing_final:
        if m in SPATIAL_VARS or m in USER_BINARY: gdf[m] = 0.0
        else: raise ValueError(f"缺少必要核心欄位：{m}")
            
    gdf = gdf.dropna(subset=[TARGET_COLUMN]).copy()
    X = gdf[FEATURE_COLUMNS].copy()
    y = gdf[TARGET_COLUMN].copy()
    
    stats: dict[str, dict[str, float]] = {}
    for col in USER_CONTINUOUS:
        if col in X.columns:
            series = pd.to_numeric(X[col], errors="coerce").fillna(0)
            q01, q99 = float(series.quantile(0.01)), float(series.quantile(0.99))
            if np.isclose(q01, q99): q01, q99 = float(series.min()), float(series.max())
            stats[col] = {"min": round(q01, 2), "max": round(q99, 2), "median": round(float(series.median()), 2), "step": 1.0 if series.dtype == int else 0.1}
            
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
    
    print("🚀 訓練 LightGBM (抹除所有標籤)...")
    model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
    
    # 訓練時不傳入 categorical_feature
    model.fit(X_train, y_train)
    
    # 🌟 關鍵：強制抹除 Booster 內部的所有元數據標籤
    booster = model.booster_
    # 這裡我們強行將該欄位的類型改為連續數值（去除 category 特性）
    booster.params.pop('categorical_feature', None)
    
    pred = model.predict(X_test)
    metrics = _metrics(y_test, pred)
    
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_stats": stats,
    }
    joblib.dump(bundle, artifact_path)
    print("✅ 訓練完成！", metrics)
    return bundle

@lru_cache(maxsize=1)
def load_or_train_bundle() -> dict[str, Any]:
    if MODEL_ARTIFACT_PATH.exists(): return joblib.load(MODEL_ARTIFACT_PATH)
    return train_and_save_model(MODEL_ARTIFACT_PATH)

def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame([{"Feature": col, "說明": FEATURE_DESCRIPTIONS.get(col, "")} for col in FEATURE_COLUMNS])

def predict_rent_per_ping(
    bundle: dict[str, Any], user_inputs: dict[str, float], location_features: dict[str, float], x: float, y: float
) -> dict[str, Any]:
    
    # 1. 空間定位
    town_gdf = _get_town_gdf()
    point = Point(x, y)
    match = town_gdf[town_gdf.contains(point)]
    town_name = "西屯區"
    if not match.empty:
        for col in ['town', 'TOWNNAME', 'TOWN', 'TOWN_NAME', 'townname', '鄉鎮市區', '鄉鎮名稱']:
            if col in match.columns:
                town_name = match.iloc[0][col]
                break
                
    # 2. 準備輸入特徵 (確保全部為 float)
    row_dict = {}
    row_dict.update({key: float(value) for key, value in user_inputs.items() if key != 'town' and key != 'transaction_year'})
    row_dict.update({key: float(value) for key, value in location_features.items()})
    
    # 強制使用整數映射
    row_dict['town'] = float(TOWN_MAPPING.get(town_name, 0))
    row_dict['transaction_year'] = float(datetime.datetime.now().year)
    
    # 補足所有 FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:
        if col not in row_dict: row_dict[col] = 0.0
            
    # 3. 🌟 終極脫殼：轉成純 NumPy Array，完全不帶任何 Pandas 標籤或 Category 屬性
    # 這是 LightGBM 預測時最穩定的格式
    input_data = np.array([[row_dict[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    
    # 使用 Booster 預測，徹底繞過 sklearn 封裝可能帶來的驗證問題
    ln_pred = float(bundle["model"].booster_.predict(input_data)[0])
    
    return {
        "ln_u_price": ln_pred, 
        "price_per_ping": float(np.exp(ln_pred)), 
        "raw_features": row_dict, 
        "detected_town": town_name
    }