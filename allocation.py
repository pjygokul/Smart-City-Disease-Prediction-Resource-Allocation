"""
allocation.py
Computes per-zone urgency scores and dynamically allocates
ICU beds, doctors, and oxygen units proportionally.
"""

import numpy as np
import pandas as pd
from data_generator import ZONES


# ── Urgency score weights ──────────────────────────────────────────────────────
W_PRED_CASES   = 0.35
W_GROWTH_RATE  = 0.30
W_CASE_RATE    = 0.20
W_RESOURCE_GAP = 0.15


def compute_growth_rate(hist_df: pd.DataFrame, zone: str, window: int = 7) -> float:
    """
    Week-over-week growth rate for a zone.
    Stored as a decimal fraction: 0.10 = 10% growth.
    Clamped to [-1.0, +5.0] to prevent SIR outbreak spikes
    from producing astronomically large percentages in the table.
    """
    z = hist_df[hist_df["zone"] == zone].sort_values("date")
    if len(z) < window * 2:
        return 0.0
    recent = z["cases"].iloc[-window:].mean()
    prev   = z["cases"].iloc[-window * 2: -window].mean()
    if prev < 1.0:          # avoid division by near-zero
        return 0.0
    raw = (recent - prev) / prev
    return float(np.clip(raw, -1.0, 5.0))   # cap at +500% / -100%


def compute_urgency(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    zones_list = list(ZONES.keys())
    records = []

    fc_7 = (
        forecast_df.sort_values("date")
        .groupby("zone")
        .apply(lambda g: g.head(7)["cases"].mean())
        .rename("pred_cases")
        .reset_index()
    )

    for zone in zones_list:
        meta = ZONES[zone]
        pop  = meta["pop"]

        pred_cases = float(fc_7.loc[fc_7["zone"] == zone, "pred_cases"].values[0]) \
                     if zone in fc_7["zone"].values else 0.0

        growth    = compute_growth_rate(hist_df, zone)
        case_rate = (pred_cases / pop) * 10_000

        icu_needed   = max(1, pred_cases * 0.02)
        resource_gap = max(0, icu_needed - meta["icu"]) / max(1, meta["icu"])

        records.append({
            "zone":             zone,
            "lat":              meta["lat"],
            "lon":              meta["lon"],
            "population":       pop,
            "pred_cases":       pred_cases,
            "growth_rate":      growth,       # decimal fraction, e.g. 0.10 = 10%
            "case_rate":        case_rate,
            "resource_gap":     resource_gap,
            "icu_capacity":     meta["icu"],
            "doctors_capacity": meta["doc"],
            "oxygen_capacity":  meta["o2"],
        })

    df = pd.DataFrame(records)

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["n_pred"]      = norm(df["pred_cases"])
    df["n_growth"]    = norm(df["growth_rate"].clip(lower=0))
    df["n_case_rate"] = norm(df["case_rate"])
    df["n_res_gap"]   = norm(df["resource_gap"])

    df["urgency_score"] = (
        W_PRED_CASES   * df["n_pred"]      +
        W_GROWTH_RATE  * df["n_growth"]    +
        W_CASE_RATE    * df["n_case_rate"] +
        W_RESOURCE_GAP * df["n_res_gap"]
    ).round(4)

    df["priority"] = pd.cut(
        df["urgency_score"],
        bins=[-1, 0.33, 0.66, 2.0],
        labels=["Low", "Medium", "High"],
    )

    return df.sort_values("urgency_score", ascending=False).reset_index(drop=True)


def allocate_resources(urgency_df: pd.DataFrame, total_icu: int, total_docs: int, total_o2: int) -> pd.DataFrame:
    df = urgency_df.copy()
    total_urgency = df["urgency_score"].sum()

    resource_totals = {
        "icu_beds": total_icu,
        "doctors": total_docs,
        "oxygen_units": total_o2
    }

    for resource, total in resource_totals.items():
        col = f"alloc_{resource}"
        if total_urgency > 0:
            df[col] = ((df["urgency_score"] / total_urgency) * total).round(1)
        else:
            df[col] = 0.0

    for resource, total in resource_totals.items():
        col  = f"alloc_{resource}"
        diff = total - df[col].sum()
        if len(df) > 0:
            df.loc[0, col] += round(diff, 1)

    return df


def get_xai_explanation(zone: str, urgency_df: pd.DataFrame,
                        feat_imp_df: pd.DataFrame) -> dict:
    row = urgency_df[urgency_df["zone"] == zone].iloc[0]

    priority_val = row["priority"]
    if hasattr(priority_val, "item"):
        priority_val = priority_val.item()
    priority_str = str(priority_val) if priority_val is not None else "Medium"

    drivers = {
        "Predicted Cases":       round(float(W_PRED_CASES   * row["n_pred"])      * 100, 1),
        "Growth Rate":           round(float(W_GROWTH_RATE  * row["n_growth"])    * 100, 1),
        "Case Rate per 10k pop": round(float(W_CASE_RATE    * row["n_case_rate"]) * 100, 1),
        "Resource Gap":          round(float(W_RESOURCE_GAP * row["n_res_gap"])   * 100, 1),
    }

    waterfall = {
        "Predicted Cases":       float(W_PRED_CASES   * row["n_pred"]),
        "Growth Rate":           float(W_GROWTH_RATE  * row["n_growth"]),
        "Case Rate":             float(W_CASE_RATE    * row["n_case_rate"]),
        "Resource Gap":          float(W_RESOURCE_GAP * row["n_res_gap"]),
    }

    top_feats = (
        feat_imp_df[feat_imp_df["zone"] == zone]
        .sort_values("importance", ascending=False)
        .head(5)[["feature", "importance"]]
        .to_dict("records")
    )

    return {
        "zone":          zone,
        "urgency_score": float(row["urgency_score"]),
        "priority":      priority_str,
        "pred_cases":    round(float(row["pred_cases"]), 1),
        # growth_rate stored as decimal → multiply by 100 for display
        "growth_rate":   round(float(np.clip(row["growth_rate"], -1.0, 5.0)) * 100, 1),
        "case_rate":     round(float(row["case_rate"]), 2),
        "drivers":       drivers,
        "waterfall":     waterfall,
        "top_features":  top_feats,
    }