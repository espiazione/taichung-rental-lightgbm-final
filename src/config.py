from __future__ import annotations
from pathlib import Path

# === 檔案路徑設定 ===
APP_ROOT = Path(__file__).resolve().parents[1]
# 這是你剛剛換上的新心臟！
DATA_PATH = APP_ROOT / "Taichung_housing_FINAL.gpkg"
POI_DIR = APP_ROOT / "POIs"
ARTIFACT_DIR = APP_ROOT / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "lightgbm_rent_model.joblib"
SPATIAL_CACHE_PATH = ARTIFACT_DIR / "spatial_cache.joblib"

EPSG_MODEL = 3826
EPSG_WEB = 4326
SEED = 42
ML_N = 30_000
KNN_K = 20

# === 🌟 你的專屬目標變數 (這樣 model.py 就能自動抓取) ===
TARGET_COLUMN = "ln_u_price"

# === 🌟 你的專屬特徵分類 ===
# 1. 使用者可在網頁輸入的「數值變數」
USER_CONTINUOUS = ["house_age", "floor_ratio", "ln_B_area"]

# 2. 使用者可在網頁勾選的「類別/虛擬變數」
USER_BINARY = [
    "is_top_floor", 
    "b_type_透天厝", 
    "b_type_公寓(5樓含以下無電梯)", 
    "transaction sign_房地(土地+建物)+車位"
]

# 3. 空間計算變數 (由 spatial.py 自動在地圖背後計算)
SPATIAL_VARS = [
    "log_dist_to_mrt_road", "log_dist_to_park_road", "log_dist_to_school_road",
    "log_dist_to_interchange_road", "log_dist_to_ghost_road", "log_dist_to_bigstore_road",
    "park_count_800m", "px_count_800m",
    "in_梧棲區三民社會住宅_net_2000", "in_豐原安康一期_net_2000", 
    "in_太平長億社會住宅_net_2000", "in_大里區光正一期社會住宅_net_2000",
    "in_烏日區高鐵社會住宅_net_2000", "in_南屯區精密機械科技創新園區社會住宅_net_2000",
    "in_東區恊園_net_2000"
]

# 將所有變數合併，這就是餵給 LightGBM 訓練的最終清單
FEATURE_COLUMNS = USER_CONTINUOUS + USER_BINARY + SPATIAL_VARS

# === 你的專屬 POI 設定 (自動抓取圖層版) ===
POI_LAYERS = {
    "mrt": (POI_DIR / "Taichung_MRT.gpkg", "Taichung_MRT"),
    "park": (POI_DIR / "Taichung_parks.gpkg", "parks"),
    "school": (POI_DIR / "Taichung_schools.gpkg", "schools"),
    "towns": (POI_DIR / "taichung_town_joined_2.gpkg", "taichung_town_joined_2"),
    "roads": (POI_DIR / "112Taichung_road_network.gpkg", "112Taichung_road_network"),
    
    "ghost": (POI_DIR / "ghost.gpkg", None),
    "pxmart": (POI_DIR / "px.gpkg", None),
    "shop": (POI_DIR / "shop.gpkg", None),
    "socialhouse": (POI_DIR / "socialhouse.gpkg", None),
    "bigstore": (POI_DIR / "bigstore.gpkg", None),
    "highway": (POI_DIR / "台中國道.gpkg", None),
}

# === 你的專屬道路距離對照表 ===
ROAD_DISTANCE_FEATURES = {
    "mrt": "log_dist_to_mrt_road",
    "park": "log_dist_to_park_road",
    "school": "log_dist_to_school_road",
    "highway": "log_dist_to_interchange_road",
    "ghost": "log_dist_to_ghost_road",
    "bigstore": "log_dist_to_bigstore_road"
}

CORE_TOWNS = {"東區", "西區", "南區", "北區", "中區", "西屯區", "北屯區", "南屯區"}

# === 🌟 你的專屬特徵說明字典 ===
# 這會在網頁首頁的表格中顯示給評審或使用者看
FEATURE_DESCRIPTIONS = {
    "house_age": "房屋屋齡 (年)。",
    "floor_ratio": "所在樓層佔總樓層的比例，數值介於 0~1 之間。",
    "ln_B_area": "建物面積 (坪數) 取自然對數。",
    "is_top_floor": "是否為頂樓，1=是，0=否。",
    "b_type_透天厝": "建物型態是否為透天厝，1=是，0=否。",
    "b_type_公寓(5樓含以下無電梯)": "建物型態是否為老舊公寓，1=是，0=否。",
    "transaction sign_房地(土地+建物)+車位": "交易內容是否包含車位，1=是，0=否。",
    "log_dist_to_mrt_road": "至最近捷運站出口的實際道路距離取自然對數。",
    "log_dist_to_park_road": "至最近公園的實際道路距離取自然對數。",
    "log_dist_to_school_road": "至最近學校的實際道路距離取自然對數。",
    "log_dist_to_interchange_road": "至最近國道交流道的實際道路距離取自然對數。",
    "log_dist_to_ghost_road": "至最近福地/嫌惡設施的實際道路距離取自然對數。",
    "log_dist_to_bigstore_road": "至最近大型量販店的實際道路距離取自然對數。",
    "park_count_800m": "目標座標 800 公尺環域內的公園數量。",
    "px_count_800m": "目標座標 800 公尺環域內的全聯數量。",
    "in_梧棲區三民社會住宅_net_2000": "是否落在梧棲三民社宅 2000m 影響範圍內。",
    "in_豐原安康一期_net_2000": "是否落在豐原安康一期社宅 2000m 影響範圍內。",
    "in_太平長億社會住宅_net_2000": "是否落在太平長億社宅 2000m 影響範圍內。",
    "in_大里區光正一期社會住宅_net_2000": "是否落在大里光正一期社宅 2000m 影響範圍內。",
    "in_烏日區高鐵社會住宅_net_2000": "是否落在烏日高鐵社宅 2000m 影響範圍內。",
    "in_南屯區精密機械科技創新園區社會住宅_net_2000": "是否落在南屯精科社宅 2000m 影響範圍內。",
    "in_東區恊園_net_2000": "是否落在東區恊園社宅 2000m 影響範圍內。"
}