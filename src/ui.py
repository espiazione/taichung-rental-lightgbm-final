from __future__ import annotations

import html
import numpy as np
import pandas as pd
import solara
import solara.lab

from .config import FEATURE_DESCRIPTIONS, CORE_TOWNS
from .map_view import MAP_CENTER, create_leafmap_widget
from .model import load_or_train_bundle, predict_rent_per_ping
from .spatial import compute_location_features, load_or_build_spatial_cache

target_lat = solara.reactive(float(MAP_CENTER[0]))
target_lon = solara.reactive(float(MAP_CENTER[1]))
selected_tab = solara.reactive(0)

APP_CSS = """
:root {
  --ds-ink: #16324f;
  --ds-ink-soft: #385169;
  --ds-green: #0f766e;
  --ds-green-dark: #14532d;
  --ds-price: #b91c1c;
  --ds-price-soft: #fee2e2;
  --ds-paper: #f7faf8;
  --ds-line: #dbe4df;
  --ds-line-strong: #b7c8c0;
}
.codex-app-page {
  width: 100%;
  box-sizing: border-box;
  padding: 22px 28px 30px 28px;
  background: linear-gradient(180deg, #f7faf8 0%, #ffffff 58%);
}
.app-panel {
  border: 1px solid var(--ds-line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 10px 28px rgba(20, 83, 45, 0.06);
  box-sizing: border-box;
  width: 100%;
  overflow: hidden;
}
.panel-body {
  padding: 16px 18px;
}
.panel-kicker {
  color: var(--ds-green);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 6px;
}
.panel-title {
  color: var(--ds-ink);
  font-size: 22px;
  line-height: 1.25;
  font-weight: 800;
  margin: 0 0 10px 0;
}
.panel-title.small {
  font-size: 18px;
  margin-bottom: 6px;
}
.panel-subtitle,
.panel-copy {
  color: var(--ds-ink-soft);
  font-size: 14px;
  line-height: 1.7;
  margin: 0;
}
.home-grid {
  display: grid;
  grid-template-columns: minmax(620px, 1.08fr) minmax(620px, 1fr);
  gap: 18px;
  align-items: stretch;
  width: 100%;
}
.home-stack {
  display: grid;
  gap: 18px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(118px, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.metric-tile {
  border: 1px solid var(--ds-line);
  border-radius: 8px;
  padding: 13px 14px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
  min-height: 78px;
}
.metric-label {
  color: #607085;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 5px;
}
.metric-value {
  color: var(--ds-ink);
  font-size: 26px;
  line-height: 1.1;
  font-weight: 850;
}
.metric-tile:first-child .metric-value {
  color: var(--ds-price);
}
.app-table-wrap {
  overflow: auto;
  padding-right: 4px;
}
.app-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  line-height: 1.45;
}
.app-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #eef6f3;
  color: #123044;
  border-bottom: 1px solid var(--ds-line-strong);
  padding: 8px 9px;
  text-align: left;
  white-space: nowrap;
}
.app-table td {
  border-bottom: 1px solid #edf2ef;
  padding: 7px 9px;
  vertical-align: top;
}
.app-table tbody tr:nth-child(even) {
  background: #f8fbfa;
}
.app-table tbody tr:hover {
  background: #fff1f1;
}
.map-page {
  padding: 14px 18px 18px 18px;
}
.map-shell {
  border: 1px solid var(--ds-line);
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 10px 28px rgba(20, 83, 45, 0.06);
  position: relative;
}
.map-badge {
  position: absolute;
  top: 14px;
  left: 62px;
  z-index: 500;
  max-width: min(420px, calc(100% - 76px));
  border: 1px solid rgba(183, 200, 192, 0.9);
  border-left: 5px solid var(--ds-price);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 10px 24px rgba(22, 50, 79, 0.12);
  padding: 10px 12px 11px 12px;
  pointer-events: none;
}
.map-badge-title {
  color: var(--ds-ink);
  font-size: 16px;
  font-weight: 850;
  line-height: 1.25;
}
.map-badge-copy {
  color: var(--ds-ink-soft);
  display: block;
  font-size: 12px;
  line-height: 1.45;
  margin-top: 4px;
}
.control-shell {
  border: 1px solid var(--ds-line);
  border-top: 4px solid var(--ds-green);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.97);
  box-shadow: 0 10px 28px rgba(20, 83, 45, 0.06);
  padding: 16px 18px 18px 18px;
  box-sizing: border-box;
}
.control-hero {
  padding-bottom: 12px;
}
.control-hero-title {
  color: var(--ds-ink);
  font-size: 20px;
  line-height: 1.25;
  font-weight: 850;
  margin: 0 0 8px 0;
}
.control-section {
  border-top: 1px solid #edf2ef;
  padding-top: 12px;
  margin-top: 12px;
}
.section-label {
  color: var(--ds-ink);
  font-size: 15px;
  font-weight: 800;
  margin-bottom: 8px;
}
.coordinate-pill {
  color: var(--ds-green-dark);
  background: #ecfdf5;
  border: 1px solid #bbf7d0;
  border-radius: 999px;
  display: inline-block;
  font-size: 13px;
  font-weight: 700;
  padding: 5px 10px;
}
.result-card {
  border: 1px solid #fecaca;
  border-left: 5px solid var(--ds-price);
  border-radius: 8px;
  background: #fff7f7;
  padding: 12px 14px;
  margin-top: 14px;
}
.result-value {
  color: var(--ds-price);
  font-size: 26px;
  font-weight: 850;
  line-height: 1.15;
}
@media (max-width: 1360px) {
  .home-grid {
    grid-template-columns: 1fr;
  }
  .metric-grid {
    grid-template-columns: repeat(2, minmax(140px, 1fr));
  }
}
@media (max-width: 1180px) {
  .map-page {
    flex-direction: column !important;
    overflow: visible !important;
  }
  .map-column,
  .control-column {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    max-width: none !important;
    width: 100% !important;
  }
  .control-column {
    max-height: none !important;
    overflow-y: visible !important;
  }
}
"""

def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(list(FEATURE_DESCRIPTIONS.items()), columns=["特徵名稱", "說明"])

def _display_table(df: pd.DataFrame, max_rows: int | None = None):
    table = df if max_rows is None else df.head(max_rows)
    solara.display(table.reset_index(drop=True))

def _table_html(df: pd.DataFrame) -> str:
    return df.reset_index(drop=True).to_html(index=False, border=0, classes="app-table", escape=True)

def _panel_html(title: str, body: str, kicker: str = "", subtitle: str = "") -> str:
    kicker_html = f"<div class='panel-kicker'>{html.escape(kicker)}</div>" if kicker else ""
    subtitle_html = f"<p class='panel-subtitle'>{html.escape(subtitle)}</p>" if subtitle else ""
    return (
        "<section class='app-panel'><div class='panel-body'>"
        f"{kicker_html}<h2 class='panel-title small'>{html.escape(title)}</h2>"
        f"{subtitle_html}{body}</div></section>"
    )

def _table_panel_html(title: str, df: pd.DataFrame, height: str, kicker: str = "", subtitle: str = "") -> str:
    table = _table_html(df)
    body = f"<div class='app-table-wrap' style='height:{height}'>{table}</div>"
    return _panel_html(title, body, kicker=kicker, subtitle=subtitle)

def _metrics_panel_html(metrics: dict, trained_rows: int) -> str:
    labels = [("R2", "R2"), ("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE_pct", "MAPE %")]
    tiles = "".join(
        f"<div class='metric-tile'><div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(str(metrics.get(key, '-')))}</div></div>"
        for key, label in labels
    )
    body = f"<div class='metric-grid'>{tiles}</div><p class='panel-subtitle' style='margin-top:12px'>訓練樣本數：{trained_rows:,} 筆實價登錄資料</p>"
    return _panel_html("模型表現指標", body, kicker="MODEL PERFORMANCE")

def _home_html(bundle: dict) -> str:
    feature_df = feature_description_table()
    fi = bundle.get("feature_importance", pd.DataFrame({"Feature": [], "Importance": []})).copy()
    if not fi.empty:
        fi["說明"] = fi["Feature"].map(FEATURE_DESCRIPTIONS).fillna("-")
    
    shap_df = bundle.get("shap_importance", pd.DataFrame({"Feature": [], "Mean_Abs_SHAP": []})).copy()
    if not shap_df.empty:
        shap_df["說明"] = shap_df["Feature"].map(FEATURE_DESCRIPTIONS).fillna("-")
    
    intro = (
        "<section class='app-panel'><div class='panel-body'>"
        "<div class='panel-kicker'>TAICHUNG HOUSE PRICE PREDICTION</div>"
        "<h1 class='panel-title'>臺中市房價與社宅外部效應預測 WebApp</h1>"
        "<p class='panel-copy'>本系統整合實價登錄大數據與地理資訊，並結合 LightGBM 機器學習演算法與空間特徵工程。預測目標為 <code>ln_u_price</code>（單價對數），"
        "系統會自動將輸出結果還原為每坪單價（萬/坪）。您可以透過地圖點擊，即時評估社會住宅、交通節點與嫌惡設施等外部效應對房價的影響。</p>"
        "</div></section>"
    )
    return (
        "<div class='codex-app-page'>"
        "<div class='home-grid'>"
        f"{intro}"
        f"{_metrics_panel_html(bundle.get('metrics', {}), int(bundle.get('trained_rows', 0)))}"
        "</div>"
        "<div class='home-grid' style='margin-top:18px'>"
        f"{_table_panel_html('特徵變數說明', feature_df, '560px', kicker='FEATURE DICTIONARY', subtitle='模型使用的核心特徵與空間變數。')}"
        "<div class='home-stack'>"
        f"{_table_panel_html('LightGBM 特徵重要性', fi, '264px', kicker='FEATURE IMPORTANCE', subtitle='依 LightGBM 節點分裂次數重要性排序。')}"
        f"{_table_panel_html('SHAP 平均絕對貢獻度', shap_df, '264px', kicker='MODEL INTERPRETATION', subtitle='計算各特徵對房屋單價的平均絕對影響力。')}"
        "</div>"
        "</div>"
        "</div>"
    )

@solara.component
def HomePage(bundle: dict):
    solara.HTML(tag="div", unsafe_innerHTML=_home_html(bundle))

@solara.component
def MapPanel():
    with solara.Column(classes=["map-shell"], style={"width": "100%"}):
        map_widget = solara.use_memo(lambda: create_leafmap_widget(target_lat, target_lon), [])
        solara.display(map_widget)
        solara.HTML(
            tag="div",
            unsafe_innerHTML=(
                "<div class='map-badge'><div class='panel-kicker'>INTERACTIVE MAP</div>"
                "<div class='map-badge-title'>實價登錄點位與目標位置</div>"
                f"<span class='map-badge-copy'>目前目標座標：{target_lon.value:.6f}, {target_lat.value:.6f}</span></div>"
            ),
        )

@solara.component
def ControlPanel(bundle: dict, spatial_cache: dict):
    # 🌟 建立人性直觀的響應式狀態變數
    town_state = solara.use_reactive("西屯區")
    house_age = solara.use_reactive(10.0)
    current_floor = solara.use_reactive(5.0)  # 使用者直觀輸入：目前在幾樓
    total_floor = solara.use_reactive(12.0)   # 使用者直觀輸入：總樓層有幾樓
    area_ping = solara.use_reactive(30.0)     # 使用者直觀輸入：幾坪的房子
    
    # 建物型態下拉選單
    b_type_display = solara.use_reactive("電梯大樓")
    # 是否為頂樓、車位
    is_top_floor = solara.use_reactive(False)
    has_parking = solara.use_reactive(True)

    prediction = solara.use_reactive(None)
    error = solara.use_reactive("")

    with solara.Column(classes=["control-shell"], style={"width": "100%"}):
        solara.HTML(
            tag="div",
            unsafe_innerHTML=(
                "<div class='control-hero'><div class='panel-kicker'>MAP PREDICTION</div>"
                "<h2 class='control-hero-title'>設定目標房屋並估算單價</h2>"
                "<p class='panel-copy'>在左側地圖點選目標位置，系統將即時計算該座標周邊的社宅、綠地與軌道等空間特徵。填寫下方屬性後即可估算。</p>"
                f"<span class='coordinate-pill' style='margin-top:10px'>目標座標：{target_lon.value:.6f}, {target_lat.value:.6f}</span></div>"
            ),
        )

        # 1. 行政區與建物型態選單
        with solara.Column(classes=["control-section"]):
            solara.HTML(tag="div", unsafe_innerHTML="<div class='section-label'>基本地理與建物屬性</div>")
            with solara.Row(gap="12px", style={"width": "100%"}):
                with solara.Column(style={"flex": "1"}):
                    solara.Select("選擇行政區", values=list(CORE_TOWNS) + ["其他區"], value=town_state)
                with solara.Column(style={"flex": "1"}):
                    solara.Select("選擇建物型態", values=["電梯大樓", "透天厝", "公寓(5樓含以下無電梯)"], value=b_type_display)

        # 2. 直觀數值拉桿 (坪數、樓層、屋齡)
        with solara.Column(classes=["control-section"]):
            solara.HTML(tag="div", unsafe_innerHTML="<div class='section-label'>房屋內部規格屬性</div>")
            
            solara.SliderFloat("房屋屋齡 (年)", value=house_age, min=0.0, max=60.0, step=1.0)
            solara.SliderFloat("欲預測的目標樓層 (樓)", value=current_floor, min=1.0, max=50.0, step=1.0)
            solara.SliderFloat("該建物總樓層數 (樓)", value=total_floor, min=1.0, max=50.0, step=1.0)
            solara.SliderFloat("房屋建物面積 (坪數)", value=area_ping, min=2.0, max=150.0, step=0.5)

        # 3. 虛擬變數勾選
        with solara.Column(classes=["control-section"]):
            solara.HTML(tag="div", unsafe_innerHTML="<div class='section-label'>其他附加條件</div>")
            with solara.Row():
                solara.Checkbox(label="此物件是否為該棟頂樓", value=is_top_floor)
                solara.Checkbox(label="交易內容是否包含車位", value=has_parking)

        def run_prediction():
            try:
                # 呼叫地理計算大腦
                location = compute_location_features(target_lon.value, target_lat.value, spatial_cache)
                
                # 🌟 幕後黑魔法：在背景將直觀數字轉換為模型需要的對數值與比例
                # A. 樓層比例 = 目標樓層 / 總樓層 (限制在 0~1 之間)
                computed_floor_ratio = float(current_floor.value / total_floor.value) if total_floor.value > 0 else 0.5
                computed_floor_ratio = min(max(computed_floor_ratio, 0.0), 1.0)
                
                # B. 坪數轉自然對數 (ln)
                computed_ln_B_area = float(np.log(area_ping.value)) if area_ping.value > 0 else 0.0
                
                # C. 下拉選單轉 Dummy 0 與 1
                is_tou_tian = 1.0 if b_type_display.value == "透天厝" else 0.0
                is_gong_yu = 1.0 if b_type_display.value == "公寓(5樓含以下無電梯)" else 0.0
                
                # 組裝給 LightGBM 的特徵字典
                user_inputs = {
                    "town": town_state.value,
                    "house_age": float(house_age.value),
                    "floor_ratio": computed_floor_ratio,
                    "ln_B_area": computed_ln_B_area,
                    "is_top_floor": 1.0 if is_top_floor.value else 0.0,
                    "b_type_透天厝": is_tou_tian,
                    "b_type_公寓(5樓含以下無電梯)": is_gong_yu,
                    "transaction sign_房地(土地+建物)+車位": 1.0 if has_parking.value else 0.0
                }
                
                result = predict_rent_per_ping(
                    bundle,
                    user_inputs,
                    location["features"],
                    float(location["details"]["x3826"]),
                    float(location["details"]["y3826"]),
                )
                prediction.value = {"prediction": result, "location": location}
                error.value = ""
            except Exception as exc:
                prediction.value = None
                error.value = f"運算錯誤: {str(exc)}"

        solara.Button("計算預測房屋單價", on_click=run_prediction, color="primary", outlined=False)

        if error.value:
            solara.Error(error.value)

        if prediction.value:
            pred = prediction.value["prediction"]
            loc = prediction.value["location"]
            
            raw_price_per_ping = pred.get("price_per_ping", 0)
            val_display_ten_k = raw_price_per_ping / 10000.0
            ln_display = pred.get("ln_u_price", 0)
            
            solara.HTML(
                tag="div",
                unsafe_innerHTML=(
                    "<div class='result-card'><div class='panel-kicker'>PREDICTED PRICE</div>"
                    f"<div class='result-value'>{val_display_ten_k:,.1f} 萬 / 坪</div>"
                    f"<p class='panel-subtitle'>模型輸出單價對數值 ln_u_price = {ln_display:.4f}</p></div>"
                ),
            )
            loc_df = pd.DataFrame([loc["details"]]).T.reset_index()
            loc_df.columns = ["區位計算項目", "實測值"]
            _display_table(loc_df)

@solara.component
def PredictionPage(bundle: dict, spatial_cache: dict):
    with solara.Row(classes=["map-page"], style={"align-items": "stretch", "flex-wrap": "wrap"}):
        with solara.Column(classes=["map-column"], style={"flex": "1 1 55%", "min-width": "0"}):
            MapPanel()
        with solara.Column(classes=["control-column"], style={"flex": "1 1 45%", "min-width": "0"}):
            ControlPanel(bundle, spatial_cache)

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
        HomePage(bundle)
    else:
        PredictionPage(bundle, spatial_cache)