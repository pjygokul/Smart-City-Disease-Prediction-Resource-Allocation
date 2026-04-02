# 🦠 Chennai Influenza Surveillance Dashboard

An **AI-powered smart city healthcare dashboard** for monitoring, forecasting, and managing Influenza spread across Chennai's 12 administrative zones — with dynamic resource allocation and Explainable AI.

---

## 🚀 Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
pydeck>=0.9.0
matplotlib>=3.7.0
```

---

## 🏗️ Project Architecture

```
chennai_flu_dashboard/
├── app.py               # Streamlit dashboard (entry point)
├── data_generator.py    # Synthetic data simulation module
├── ai_model.py          # GBM forecasting + feature engineering
├── allocation.py        # Urgency scoring + resource allocation
├── visualizations.py    # Plotly chart builders
├── map_layer.py         # PyDeck geospatial layer
└── requirements.txt
```

### Module Responsibilities

| Module | Purpose |
|---|---|
| `data_generator.py` | Generates 180-day synthetic flu case data per zone with Chennai's seasonal patterns (NE monsoon + summer) |
| `ai_model.py` | Trains a Gradient Boosting Regressor per zone on 13 engineered time-series features; iterative 14-day forecast |
| `allocation.py` | Computes weighted urgency scores and distributes ICU beds, doctors, oxygen proportionally |
| `visualizations.py` | All Plotly figures: trend lines, bar charts, urgency waterfall, feature importance |
| `map_layer.py` | PyDeck ScatterplotLayer coloured by priority tier, sized by urgency score |
| `app.py` | Streamlit UI: sidebar controls, KPI cards, all charts, XAI panel, allocation table |

---

## 🤖 AI Model Details

**Algorithm**: `GradientBoostingRegressor` (sklearn)

**Features** (13 total):
- Temporal: day of year, day of week, week number, month
- Seasonal: `sin(2π·doy/365)`, `cos(2π·doy/365)` — captures cyclical patterns
- Lag features: 1-day, 3-day, 7-day, 14-day lag of case counts
- Rolling stats: 7-day rolling mean, 14-day rolling mean, 7-day rolling std

**Training**: One model per zone, fit on full historical window.

**Inference**: Iterative — each future day's prediction is appended to history before predicting the next.

---

## 🎯 Urgency Score Formula

```
Urgency = 0.35 × norm(predicted_cases)
        + 0.30 × norm(growth_rate)       ← week-over-week growth, clipped at 0
        + 0.20 × norm(case_rate)         ← cases per 10,000 population
        + 0.15 × norm(resource_gap)      ← ICU deficit vs projected need
```

All components are min-max normalised to [0,1] across zones before weighting.

**Priority tiers**: Low (0–0.33) · Medium (0.33–0.66) · High (0.66–1.0)

---

## 📊 Resource Allocation Logic

Total city-wide resources are distributed **proportionally by urgency score**:

```python
alloc_zone = (urgency_zone / sum_urgency) × total_resource
```

Resources allocated:
- **ICU Beds** — total: 618
- **Doctors** — total: 309
- **Oxygen Units** — total: 1,155

---

## 🧠 Explainable AI (XAI)

For any selected zone the dashboard shows:

1. **Driver breakdown** — % contribution of each urgency component
2. **Model feature importance** — top 8 GBM features by split importance
3. **Zone fact card** — urgency score, priority, predicted cases, growth %, case rate

---

## 🗺️ Zones Covered

| Zone | Population | ICU Beds |
|---|---|---|
| Tondiarpet | 195,000 | 42 |
| Royapuram | 170,000 | 35 |
| Anna Nagar | 220,000 | 78 |
| Adyar | 205,000 | 65 |
| Tambaram | 240,000 | 55 |
| Velachery | 190,000 | 60 |
| Perambur | 175,000 | 40 |
| Kodambakkam | 185,000 | 50 |
| Sholinganallur | 160,000 | 45 |
| Ambattur | 210,000 | 48 |
| Mylapore | 155,000 | 70 |
| Manali | 145,000 | 30 |

---

## 📈 Scalability Notes

- **Real data**: Replace `generate_historical_data()` in `data_generator.py` with a database query or API call returning the same `(date, zone, cases)` DataFrame format.
- **More zones**: Add entries to `ZONES` dict in `data_generator.py`; everything else scales automatically.
- **Live updates**: Wrap `run_pipeline()` in a `st.fragment` with `run_every` for streaming updates.
- **More resources**: Add new keys to `RESOURCE_TOTALS` and corresponding columns in `ZONES`.

---

## 🎨 Design

Dark medical theme with `Space Grotesk` + `JetBrains Mono` typography.  
Color semantics: 🔴 High risk · 🟡 Medium risk · 🟢 Low risk.
