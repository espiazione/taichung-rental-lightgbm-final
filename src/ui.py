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

target_lat = solara.reactive(float(MAP_CENTER[0]))
target_lon = solara.reactive(float(MAP_CENTER[1]))
selected_tab = solara.reactive(0)

# --- 你的 CSS 樣式 ---
APP_CSS = """.codex-app-page { width: 100%; box-sizing: border-box; padding: 22px; } .app-panel { border: 1px solid #dbe4df; border-radius: 8px; }"""

# --- 輔助函式區 ---
def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(list(FEATURE_DESCRIPTIONS.items()), columns=["特徵名稱", "說明"])

def _yes_no(value: str) -> float:
    return 1.0 if value == "是" else 0.0

def _display_table(df: pd.DataFrame):
    solara.display(df.reset_index(drop=True))

def _table_html(df: pd.DataFrame) -> str:
    return df.reset_index(drop=True).to_html(index=False, border=0, classes="app-table")

def _panel_html(title: str, body: str, kicker: str = "") -> str:
    return f"<section class='app-panel'><div class='panel-body'><div class='panel-kicker'>{kicker}</div><h2>{title}</h2>{body}</div></section>"

def _home_html(bundle: dict) -> str:
    feature_df = feature_description_table()
    fi = bundle.get("feature_importance", pd.DataFrame({"Feature": [], "Importance": []})).copy()
    if not fi.empty:
        fi["說明"] = fi["Feature"].map(FEATURE_DESCRIPTIONS).fillna("-")
    
    shap_df = bundle.get("shap_importance", pd.DataFrame({"Feature": [], "Mean_Abs_SHAP": []})).copy()
    if not shap_df.empty:
        shap_df["說明"] = shap_df["Feature"].map(FEATURE_DESCRIPTIONS).fillna("-")
    
    # 簡化首頁 HTML 邏輯以確保格式正確
    return (
        f"<div class='codex-app-page'><h1>臺中市房價預測 WebApp</h1>"
        f"<h3>模型指標</h3>{_table_html(pd.DataFrame([bundle.get('metrics', {})]))}"
        f"<h3>特徵重要性</h3>{_table_html(fi.head(10))}"
        "</div>"
    )

@solara.component
def HomePage(bundle: dict):
    solara.HTML(tag="div", unsafe_innerHTML=_home_html(bundle))

@solara.component
def MapPanel():
    with solara.Column(style={"width": "100%"}):
        map_widget = solara.use_memo(lambda: create_leafmap_widget(target_lat, target_lon), [])
        solara.display(map_widget)

@solara.component
def ControlPanel(bundle: dict, spatial_cache: dict):
    # 響應式狀態
    house_age = solara.use_reactive(10.0)
    current_floor = solara.use_reactive(5.0)
    total_floor = solara.use_reactive(12.0)
    area_ping = solara.use_reactive(30.0)
    b_type_display = solara.use_reactive("電梯大樓")
    is_top_floor = solara.use_reactive(False)
    has_parking = solara.use_reactive(True)
    prediction = solara.use_reactive(None)

    with solara.Column():
        solara.Select("選擇建物型態", values=["電梯大樓", "透天厝", "公寓(5樓含以下無電梯)"], value=b_type_display)
        solara.SliderFloat("房屋屋齡 (年)", value=house_age, min=0.0, max=60.0, step=1.0)
        solara.SliderFloat("欲預測的目標樓層 (樓)", value=current_floor, min=1.0, max=50.0, step=1.0)
        solara.SliderFloat("該建物總樓層數 (樓)", value=total_floor, min=1.0, max=50.0, step=1.0)
        solara.SliderFloat("房屋建物面積 (坪)", value=area_ping, min=2.0, max=150.0, step=0.5)
        solara.Checkbox(label="此物件是否為該棟頂樓", value=is_top_floor)
        solara.Checkbox(label="交易內容是否包含車位", value=has_parking)

        def run_prediction():
            try:
                location = compute_location_features(target_lon.value, target_lat.value, spatial_cache)
                auto_town = location.get("auto_town", "其他區")
                
                computed_floor_ratio = float(current_floor.value / total_floor.value) if total_floor.value > 0 else 0.5
                user_inputs = {
                    "town": auto_town,
                    "house_age": float(house_age.value),
                    "floor_ratio": min(max(computed_floor_ratio, 0.0), 1.0),
                    "ln_B_area": float(np.log(area_ping.value)),
                    "is_top_floor": 1.0 if is_top_floor.value else 0.0,
                    "b_type_透天厝": 1.0 if b_type_display.value == "透天厝" else 0.0,
                    "b_type_公寓(5樓含以下無電梯)": 1.0 if b_type_display.value == "公寓(5樓含以下無電梯)" else 0.0,
                    "transaction sign_房地(土地+建物)+車位": 1.0 if has_parking.value else 0.0
                }
                
                res = predict_rent_per_ping(bundle, user_inputs, location["features"], float(location["details"]["x3826"]), float(location["details"]["y3826"]))
                prediction.value = res["price_per_ping"] / 10000.0
            except Exception as e:
                prediction.value = f"錯誤: {e}"

        solara.Button("計算預測單價", on_click=run_prediction, color="primary")
        if prediction.value:
            solara.Markdown(f"### 預測單價: {prediction.value:,.1f} 萬/坪")

@solara.component
def PredictionPage(bundle: dict, spatial_cache: dict):
    with solara.Row():
        MapPanel()
        ControlPanel(bundle, spatial_cache)

@solara.component
def Page():
    # 確保 bundle 與 spatial_cache 只被呼叫一次
    bundle = solara.use_memo(load_or_train_bundle, [])
    spatial_cache = solara.use_memo(load_or_build_spatial_cache, [])
    
    solara.Title("臺中市房價與社宅外部效應預測 WebApp")
    solara.HTML(tag="style", unsafe_innerHTML=APP_CSS)
    
    # 使用正確的 Solara Tabs 語法，這會解決頁面空白問題
    with solara.lab.Tabs(grow=True, color="#0f766e", slider_color="#b91c1c"):
        with solara.lab.Tab("首頁說明"):
            HomePage(bundle)
        with solara.lab.Tab("互動地圖預測"):
            PredictionPage(bundle, spatial_cache)