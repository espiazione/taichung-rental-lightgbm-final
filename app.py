import warnings
warnings.filterwarnings("ignore")  # 順便把終端機裡那些煩人的紅字警告關掉

print("⏳ [系統啟動中] 正在將房價模型與空間快取預先載入記憶體 (約需 5~10 秒)...")

# 1. 預先引入函數
from src.model import load_or_train_bundle
from src.spatial import load_or_build_spatial_cache

# 2. 在網頁伺服器啟動前，強制執行載入！(資料會被存進 lru_cache 快取中)
_ = load_or_train_bundle()
_ = load_or_build_spatial_cache()

print("✅ [載入完成] 大腦與空間記憶已就緒！準備渲染網頁介面...")

# 3. 資料準備好後，才把網頁介面叫出來
from src.ui import Page

__all__ = ["Page"]