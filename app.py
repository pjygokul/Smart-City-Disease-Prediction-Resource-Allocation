"""
app.py  ──  Chennai Influenza Surveillance Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Chennai Flu Watch",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (medical dark theme) ───────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    min-height: 100vh;
  }
  .main-header {
    background: linear-gradient(90deg, #1a1f3a 0%, #0f1729 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .main-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    color: #94a3b8;
    margin: 6px 0 0;
    font-size: 0.95rem;
  }
  .kpi-card {
    background: linear-gradient(135deg, #131929 0%, #0f1e35 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: rgba(59,130,246,0.5); }
  .kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    display: block;
  }
  .kpi-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }
  .kpi-card.danger .kpi-value  { color: #f87171; }
  .kpi-card.warning .kpi-value { color: #fbbf24; }
  .kpi-card.success .kpi-value { color: #4ade80; }
  .section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #cbd5e1;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    margin: 28px 0 16px;
  }
  .priority-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }
  .badge-High   { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
  .badge-Medium { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
  .badge-Low    { background: rgba(34,197,94,0.2);  color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
  .xai-box {
    background: #0f1729;
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 16px;
  }
  .xai-box h4 { color: #93c5fd; margin: 0 0 12px; font-size: 0.95rem; }
  .xai-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
    color: #94a3b8;
  }
  .xai-row span:last-child { color: #60a5fa; font-weight: 600; }
  .stSelectbox > div > div { background: #0f1729 !important; }
  .stMultiSelect > div > div { background: #0f1729 !important; }
  .block-container { padding-top: 1.5rem !important; }
  div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Module imports (after page config) ────────────────────────────────────────
from data_generator import generate_historical_data, get_zone_metadata, ZONES
from ai_model       import train_all_models, generate_forecasts, get_feature_importance_df
from allocation     import compute_urgency, allocate_resources, get_xai_explanation
from visualizations import (trend_chart, resource_bar_chart, urgency_heatmap,
                             xai_waterfall, feature_importance_chart, kpi_summary)
from map_layer      import build_map


# ── Cached data pipeline ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating simulation data…")
def load_data(days: int = 180):
    return generate_historical_data(days=days)

@st.cache_resource(show_spinner="Training AI forecasting models…")
def load_models(days: int = 180):
    hist = generate_historical_data(days=days)
    return train_all_models(hist)

def run_pipeline(days: int = 180, horizon: int = 14, total_icu: int = 150, total_docs: int = 300, total_o2: int = 800):
    hist_df     = load_data(days)
    models      = load_models(days)
    forecast_df = generate_forecasts(models, horizon)
    feat_imp    = get_feature_importance_df(models)
    urgency_df  = compute_urgency(hist_df, forecast_df)
    alloc_df    = allocate_resources(urgency_df, total_icu, total_docs, total_o2)
    return hist_df, forecast_df, feat_imp, urgency_df, alloc_df, models


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    days    = st.slider("Historical window (days)", 60, 365, 180, 30)
    horizon = st.slider("Forecast horizon (days)",  7,  30,  14,  1)

    st.markdown("---")
    st.markdown("### 📦 Available Resources")
    total_icu = st.number_input("Additional ICU Beds", min_value=0, value=150, step=10)
    total_docs = st.number_input("Additional Doctors", min_value=0, value=300, step=10)
    total_o2 = st.number_input("Additional O₂ Units", min_value=0, value=800, step=50)

    st.markdown("---")
    st.markdown("### 🗺️ Zone Filter")
    all_zones = list(ZONES.keys())
    selected_zones = st.multiselect(
        "Select zones for trend chart",
        all_zones,
        default=all_zones[:5],
    )
    if not selected_zones:
        selected_zones = all_zones[:5]

    st.markdown("---")
    st.markdown("### 🔬 XAI Zone")
    xai_zone = st.selectbox("Explain priority for:", all_zones)

    st.markdown("---")
    st.caption("🦠 Chennai Flu Watch v1.0\nPowered by GBM + SHAP-style XAI")


# ── Main pipeline ──────────────────────────────────────────────────────────────
hist_df, forecast_df, feat_imp, urgency_df, alloc_df, models = run_pipeline(days, horizon, total_icu, total_docs, total_o2)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🦠 Chennai Influenza Surveillance Dashboard</h1>
  <p>AI-powered seasonal disease monitoring · Zonal risk scoring · Dynamic resource allocation</p>
</div>
""", unsafe_allow_html=True)


# ── KPI Row ───────────────────────────────────────────────────────────────────
kpi = kpi_summary(urgency_df, hist_df)
c1, c2, c3, c4, c5 = st.columns(5)

kpi_data = [
    (c1, kpi["latest_daily"],    "Latest Daily Cases",       "warning"),
    (c2, kpi["total_pred_7d"],   "Projected Cases (7d)",     "danger"),
    (c3, kpi["high_risk_zones"], "High-Risk Zones",          "danger"),
    (c4, f"{kpi['avg_growth_pct']:+.1f}%", "Avg Weekly Growth", "warning"),
    (c5, kpi["total_icu"],       "Total ICU Capacity",       "success"),
]
for col, val, label, cls in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
          <span class="kpi-value">{val}</span>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Row 1: Trend + Urgency ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Disease Trend Analysis</div>', unsafe_allow_html=True)
col_l, col_r = st.columns([3, 2])

with col_l:
    st.plotly_chart(trend_chart(hist_df, forecast_df, selected_zones),
                    use_container_width=True)

with col_r:
    st.plotly_chart(urgency_heatmap(urgency_df), use_container_width=True)


# ── Row 2: Map + Resource Allocation ─────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Geospatial Risk Map · 📦 Resource Allocation</div>',
            unsafe_allow_html=True)
col_map, col_alloc = st.columns([2, 3])

with col_map:
    deck = build_map(urgency_df)
    st.pydeck_chart(deck, use_container_width=True)
    st.caption("🔴 High · 🟡 Medium · 🟢 Low  |  Circle size = urgency score")

with col_alloc:
    st.plotly_chart(resource_bar_chart(alloc_df), use_container_width=True)


# ── Row 3: Allocation table ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Detailed Allocation Table</div>',
            unsafe_allow_html=True)

table_df = alloc_df[[
    "zone", "priority", "urgency_score",
    "pred_cases", "growth_rate",
    "alloc_icu_beds", "alloc_doctors", "alloc_oxygen_units",
    "icu_capacity", "doctors_capacity", "oxygen_capacity",
]].copy()

table_df.columns = [
    "Zone", "Priority", "Urgency", "Pred Cases (7d)",
    "Growth Rate", "Alloc ICU", "Alloc Doctors", "Alloc O₂",
    "ICU Cap", "Dr Cap", "O₂ Cap",
]
table_df["Growth Rate"] = (table_df["Growth Rate"] * 100).round(1).astype(str) + "%"
table_df["Urgency"]     = table_df["Urgency"].round(4)

def priority_style(val):
    clr = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"}.get(str(val), "white")
    return f"color: {clr}; font-weight: 600;"

styled = (
    table_df.style
    .applymap(priority_style, subset=["Priority"])
    .background_gradient(subset=["Urgency"], cmap="Reds")
    .format({"Alloc ICU": "{:.0f}", "Alloc Doctors": "{:.0f}", "Alloc O₂": "{:.0f}"})
    .set_properties(**{"background-color": "#0d1526", "color": "#cbd5e1", "border": "1px solid #1e3a5f"})
)
st.dataframe(styled, use_container_width=True, height=340)


# ── Row 4: XAI Section ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 Explainable AI — Priority Justification</div>',
            unsafe_allow_html=True)

xai = get_xai_explanation(xai_zone, alloc_df, feat_imp)
x1, x2, x3 = st.columns([2, 2, 2])

with x1:
    priority = xai["priority"]
    badge_cls = f"badge-{priority}"
    st.markdown(f"""
    <div class="xai-box">
      <h4>🎯 Zone Summary</h4>
      <div class="xai-row"><span>Zone</span><span>{xai['zone']}</span></div>
      <div class="xai-row"><span>Priority</span>
        <span><span class="priority-badge {badge_cls}">{priority}</span></span></div>
      <div class="xai-row"><span>Urgency Score</span><span>{xai['urgency_score']:.4f}</span></div>
      <div class="xai-row"><span>Predicted Cases (7d avg)</span><span>{xai['pred_cases']:.0f}</span></div>
      <div class="xai-row"><span>Weekly Growth</span><span>{xai['growth_rate']:+.1f}%</span></div>
      <div class="xai-row"><span>Case Rate (per 10k)</span><span>{xai['case_rate']:.2f}</span></div>
    </div>""", unsafe_allow_html=True)

with x2:
    st.plotly_chart(xai_waterfall(xai), use_container_width=True)

with x3:
    st.plotly_chart(feature_importance_chart(feat_imp, xai_zone), use_container_width=True)

# ── Row 5: Model MAE summary ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance (MAE per Zone)</div>',
            unsafe_allow_html=True)

mae_rows = [{"Zone": z, "MAE (cases)": round(m.mae, 2)} for z, m in models.items()]
mae_df   = pd.DataFrame(mae_rows).sort_values("MAE (cases)")
st.dataframe(
    mae_df.style
    .background_gradient(subset=["MAE (cases)"], cmap="Blues")
    .set_properties(**{"background-color": "#0d1526", "color": "#cbd5e1",
                       "border": "1px solid #1e3a5f"}),
    use_container_width=True, height=200,
)

st.markdown("---")
st.caption(
    "Chennai Flu Watch · Simulated data for demonstration · "
    "GradientBoosting Regressor with 13 time-series features · "
    "Urgency score = 0.35×predicted + 0.30×growth + 0.20×case-rate + 0.15×resource-gap"
)