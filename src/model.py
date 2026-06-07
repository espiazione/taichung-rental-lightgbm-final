# src/model.py
from __future__ import annotations
import time
from functools import lru_cache
from typing import Any
import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 匯入你剛改好的 config 變數
from .config import (
    ARTIFACT_DIR, DATA_PATH, FEATURE_COLUMNS, FEATURE_DESCRIPTIONS,
    MODEL_ARTIFACT_PATH, SEED, TARGET_COLUMN, USER_CONTINUOUS,
    USER_BINARY, SPATIAL_VARS, POI_LAYERS, CORE_TOWNS
)

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MAPE_pct": round(float(mape), 2),
    }

def _prepare_training_frame() -> tuple[pd.DataFrame, pd.Series, dict[str, dict[str, float]]]:
    # 1. 讀取空間資料表
    gdf = gpd.read_file(DATA_PATH)
    
    # 建立一個動態維護的訓練特徵清單
    # 因為我們等一下會加入 town_xxx，所以要複製一份 config 的清單來擴充
    dynamic_features = FEATURE_COLUMNS.copy()
    
    # --- 關鍵修正：確保移除可能混入的字串特徵 'town' ---
    if 'town' in dynamic_features:
        dynamic_features.remove('town')
    
    # 2. 轉換單價單位與清理極端值
    if 'u_price' in gdf.columns:
        # 將原本的 (元/平方公尺) 乘上 3.3058，轉換成 (元/坪)
        gdf['u_price_ping'] = pd.to_numeric(gdf['u_price'], errors='coerce') * 3.3058
        
        # 預先排除不可理喻的極低價 (例如低於 1,000 元/坪的異常資料)
        gdf = gdf[gdf['u_price_ping'] > 1000].copy()
        
        # 利用 IQR (四分位距法) 濾除極端天價與超低價
        Q1 = gdf['u_price_ping'].quantile(0.25)
        Q3 = gdf['u_price_ping'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = max(1000, Q1 - 1.5 * IQR) 
        
        gdf = gdf[(gdf['u_price_ping'] >= lower_bound) & (gdf['u_price_ping'] <= upper_bound)].copy()

    # 3. 處理行政區 One-Hot Encoding (產生 town_西屯區 等虛擬變數)
    town_col = next((c for c in ['township', 'town', 'TOWN'] if c in gdf.columns), None)
    town_dummies_cols = []
    if town_col:
        # 只保留指定的 CORE_TOWNS，其他的歸類為 '其他區' 或直接 drop
        gdf[town_col] = gdf[town_col].apply(lambda x: x if x in CORE_TOWNS else "其他區")
        dummies = pd.get_dummies(gdf[town_col], prefix='town').astype(float)
        gdf = pd.concat([gdf, dummies], axis=1)
        town_dummies_cols = dummies.columns.tolist()
        
        # 把展開後的虛擬變數加進我們的動態特徵清單中
        for col in town_dummies_cols:
            if col not in dynamic_features:
                dynamic_features.append(col)

    # 4. 計算對數 (目標變數 ln_u_price)
        # 現在我們取 log 的對象，已經是轉換好的「元/坪」了
        gdf['ln_u_price'] = np.log(gdf['u_price_ping'].clip(lower=1e-5)).fillna(0)
    
    # 如果 config 指定了 ln_B_area，確保我們有計算它
    if 'B_area_m2' in gdf.columns and 'ln_B_area' not in gdf.columns:
        # 假設 B_area_m2 是平方公尺，轉換為坪 (1 平方公尺 = 0.3025 坪)
        gdf['ln_B_area'] = np.log((pd.to_numeric(gdf['B_area_m2'], errors='coerce') * 0.3025).clip(lower=1e-5)).fillna(0)
    elif 'B_area' in gdf.columns and 'ln_B_area' not in gdf.columns:
         gdf['ln_B_area'] = np.log(pd.to_numeric(gdf['B_area'], errors='coerce').clip(lower=1e-5)).fillna(0)

    # 5. 二次清理目標變數異常
    if TARGET_COLUMN in gdf.columns: 
        gdf = gdf[gdf[TARGET_COLUMN] > 0] 

    # 6. 補齊可能缺漏的特徵欄位，並填補為 0
    for m in dynamic_features:
        if m not in gdf.columns: 
            gdf[m] = 0.0
            
    # 只回傳我們確定的動態特徵與目標變數
    X = gdf[dynamic_features].copy()
    y = gdf[TARGET_COLUMN].copy()
    
    # 將這份實際用來訓練的完整欄位名稱記錄在 stats 中
    stats = {"trained_columns": dynamic_features}
            
    return X, y, stats

def train_and_save_model(artifact_path=MODEL_ARTIFACT_PATH) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    X, y, stats = _prepare_training_frame()
    X = X.astype(np.float32)
    trained_columns = stats["trained_columns"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    print("🚀 訓練中 (終極穩定模式)...")
    # 使用 scikit-learn API 進行擬合，方便計算 metric
    model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=1)
    
    # 不指定 categorical_feature，因為我們已經把 town 展開成 dummies 了
    model.fit(X_train, y_train) 
    
    # 1. 存出純模型文字檔 (避免版本不相容)
    model.booster_.save_model(str(ARTIFACT_DIR / "model.txt"))
    
    pred = model.predict(X_test)
    metrics = _metrics(y_test, pred)

    print("📊 正在計算特徵重要性與 SHAP 貢獻度...")
    # --- 新增：計算特徵重要性 ---
    fi_df = pd.DataFrame({
        "Feature": trained_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    # --- 新增：計算 SHAP 貢獻度 ---
    import shap
    # 為了避免資料量太大算太久，我們抽取測試集前 5000 筆來計算 SHAP
    sample_X = X_test[:5000] if len(X_test) > 5000 else X_test
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    shap_df = pd.DataFrame({
        "Feature": trained_columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    
    # 2. 將 bundle 打包 (加入 fi_df 與 shap_df)
    bundle = {
        "feature_columns": trained_columns,
        "metrics": metrics, 
        "trained_rows": len(X),
        "feature_importance": fi_df,
        "shap_importance": shap_df
    }
    
    joblib.dump(bundle, artifact_path, compress=3)
    print(f"✅ 訓練完成，資料筆數: {len(X)}，R2: {metrics['R2']}!")
    return bundle

@lru_cache(maxsize=1)
def load_or_train_bundle() -> dict[str, Any]:
    # 這裡的邏輯是你原本很棒的讀取法
    if MODEL_ARTIFACT_PATH.exists() and (ARTIFACT_DIR / "model.txt").exists():
        bundle = joblib.load(MODEL_ARTIFACT_PATH)
        # 用純文字檔將原生模型掛載回來
        bundle["model"] = lgb.Booster(model_file=str(ARTIFACT_DIR / "model.txt"))
        return bundle
    
    return train_and_save_model(MODEL_ARTIFACT_PATH)

def predict_rent_per_ping(
    bundle: dict[str, Any], user_inputs: dict[str, Any], location_features: dict[str, float], x: float, y: float
) -> dict[str, Any]:
    
    # 1. 將使用者輸入的數值整理成 float
    row_dict = {}
    for k, v in user_inputs.items():
        if k == 'town':
            # 將使用者選擇的行政區，轉換成 dummy 變數形式
            town_key = f"town_{v}"
            row_dict[town_key] = 1.0
        else:
            row_dict[k] = float(v)
            
    # 2. 併入空間變數
    row_dict.update({k: float(v) for k, v in location_features.items()})
    
    # 3. 取得模型「真正」吃進去的特徵清單 (包含 town_西屯區 等)
    trained_columns = bundle["feature_columns"]
    
    # 4. 對齊特徵，缺少的補 0.0
    for col in trained_columns:
        if col not in row_dict: 
            row_dict[col] = 0.0
            
    # 5. 組裝最終 numpy array 進行預測
    input_data = np.array([[row_dict[col] for col in trained_columns]], dtype=np.float32)
    
    # 使用原生 .predict()
    ln_pred = float(bundle["model"].predict(input_data)[0])
    
    return {
        "ln_u_price": ln_pred, 
        "price_per_ping": float(np.exp(ln_pred))
    }

if __name__ == "__main__":
    # 如果直接執行此檔案，就重訓模型
    train_and_save_model()