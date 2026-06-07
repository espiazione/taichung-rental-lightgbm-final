from __future__ import annotations
import html
import numpy as np
import pandas as pd
import solara
import solara.lab

from .config import FEATURE_DESCRIPTIONS, CORE_TOWNS, USER_BINARY, USER_CONTINUOUS
from .map_view import MAP_CENTER, create_leafmap_widget
from .model import load_or_train_bundle, predict_rent_per_ping
from .spatial import compute_location_features, load_or_build_spatial_cache

# --- 下面這堆 CSS 保持不變，省略以節省空間 ---
APP_CSS = """...""" # (使用你原本的 CSS 即可)

# --- 這裡定義 Helper 函式 ---
def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(list(FEATURE_DESCRIPTIONS.items()), columns=["特徵名稱", "說明"])

# ... (其他的 helper: _yes_no, _display_table, _table_html, _panel_html, _table_panel_html, _metrics_panel_html 等保持不變)

# --- 這裡定義 Homepage UI ---
def _home_html(bundle: dict) -> str:
    # 這是你在首頁看到的表格邏輯
    feature_df = feature_description_table()
    fi = bundle.get("feature_importance", pd.DataFrame({"Feature": [], "Importance": []}))
    shap_df = bundle.get("shap_importance", pd.DataFrame({"Feature": [], "Mean_Abs_SHAP": []}))
    
    # 這裡放入你原本的 _home_html 內容，確保這段程式碼順序正確
    return f"<div class='codex-app-page'>...</div>"

@solara.component
def HomePage(bundle: dict):
    solara.HTML(tag="div", unsafe_innerHTML=_home_html(bundle))

# --- 這裡定義 MapPanel, ControlPanel, PredictionPage ---
@solara.component
def MapPanel():
    # ... (保持原樣) ...

@solara.component
def ControlPanel(bundle: dict, spatial_cache: dict):
    # ... (保持原樣，確保這段程式碼在 Page() 之前) ...

@solara.component
def PredictionPage(bundle: dict, spatial_cache: dict):
    # ... (保持原樣) ...

# --- 最後定義 Page()，確保它呼叫上面的元件 ---
@solara.component
def Page():
    solara.Title("臺中市房價與社宅外部效應預測 WebApp")
    solara.HTML(tag="style", unsafe_innerHTML=APP_CSS)
    
    bundle = solara.use_memo(load_or_train_bundle, [])
    spatial_cache = solara.use_memo(load_or_build_spatial_cache, [])
        
    with solara.lab.Tabs(value=selected_tab, grow=True, color="#0f766e", slider_color="#b91c1c"):
        solara.lab.Tab("首頁說明")
        solara.lab.Tab("互動地圖預測")
        
    if selected_tab.value == 0:
        HomePage(bundle) # 這裡現在一定找得到 HomePage
    else:
        PredictionPage(bundle, spatial_cache)