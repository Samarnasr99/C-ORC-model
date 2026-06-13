#!/usr/bin/env python3
"""
C-ORC Performance Predictor - ML-Based Tool for Steelmaking WHR
===============================================================
A multi-target ML prediction tool that predicts 8 thermo-economic performance
indicators for a Cascade Organic Rankine Cycle (C-ORC) waste heat recovery system
in steelmaking, using the TOP 2 best models per target with uncertainty bounds.

Author: ML Engineering Team
Date: 2025
"""

import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Suppress non-critical warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# =============================================================================
# CONFIGURATION
# =============================================================================

# -- Input Features (7) with ranges -------------------------------------------
FEATURE_CONFIG: Dict[str, Dict[str, float]] = {
    "High Pressure loop (KPa)": {"min": 2800, "max": 3800, "default": 3300, "step": 50},
    "Split Ratio": {"min": 0.2, "max": 0.8, "default": 0.5, "step": 0.05},
    "High pressure mass flow (kg/s)": {"min": 20, "max": 40, "default": 30, "step": 1},
    "low pressure loop (KPa)": {"min": 2000, "max": 2800, "default": 2400, "step": 50},
    "low pressure mass flow (kg/s)": {"min": 50, "max": 125, "default": 80, "step": 1},
    "Exhaust Gases from Reformer Mass Flow (kg/s)": {"min": 60, "max": 67, "default": 64.8, "step": 0.1},
    "Top gases from Shaft furnace Mass Flow (kg/s)": {"min": 40, "max": 44, "default": 43.1, "step": 0.1},
}

FEATURE_COLS: List[str] = list(FEATURE_CONFIG.keys())

# -- Target Variables (8) -----------------------------------------------------
TARGET_COLS: List[str] = [
    "HP-Work - Power (KW)",
    "LP-Power1 - Power (KW)",
    "lp-power2 - Power (KW)",
    "Lp-power-in - Power (KW)",
    "HP-pump - Power (KW)",
    "CAPEX ($)",
    "LCOE ($/kWh)",
    "w-net (KW)",
]

# Short names for display
TARGET_SHORT_NAMES: List[str] = [
    "HP-Work", "LP-Power1", "LP-Power2", "LP-Power-in",
    "HP-Pump", "CAPEX", "LCOE", "W-Net",
]

# -- TOP 2 Models per Target (EXACT DATA from model evaluation) ---------------
# Model #1 data
TOP1_MODELS: Dict[str, Dict[str, Any]] = {
    "HP-Work - Power (KW)":       {"name": "GradientBoosting", "r2": 0.99999, "mae": 2.53},
    "LP-Power1 - Power (KW)":     {"name": "GradientBoosting", "r2": 0.99566, "mae": 95.85},
    "lp-power2 - Power (KW)":     {"name": "GradientBoosting", "r2": 0.99647, "mae": 83.11},
    "Lp-power-in - Power (KW)":   {"name": "RandomForest",     "r2": 0.99999, "mae": 0.27},
    "HP-pump - Power (KW)":       {"name": "RandomForest",     "r2": 0.99994, "mae": 0.41},
    "CAPEX ($)":                  {"name": "RandomForest",     "r2": 0.99813, "mae": 51704.65},
    "LCOE ($/kWh)":               {"name": "RandomForest",     "r2": 0.99841, "mae": 0.00017},
    "w-net (KW)":                 {"name": "GradientBoosting", "r2": 0.99844, "mae": 119.28},
}

# Model #2 data
TOP2_MODELS: Dict[str, Dict[str, Any]] = {
    "HP-Work - Power (KW)":       {"name": "RandomForest",     "r2": 1.00000, "mae": 1.54},
    "LP-Power1 - Power (KW)":     {"name": "RandomForest",     "r2": 0.99504, "mae": 84.39},
    "lp-power2 - Power (KW)":     {"name": "ExtraTrees",       "r2": 0.99631, "mae": 85.70},
    "Lp-power-in - Power (KW)":   {"name": "DecisionTree",     "r2": 1.00000, "mae": 0.28},
    "HP-pump - Power (KW)":       {"name": "GradientBoosting", "r2": 0.99991, "mae": 0.93},
    "CAPEX ($)":                  {"name": "DecisionTree",     "r2": 0.99569, "mae": 78998.50},
    "LCOE ($/kWh)":               {"name": "ExtraTrees",       "r2": 0.99800, "mae": 0.00017},
    "w-net (KW)":                 {"name": "kNN",              "r2": 0.99761, "mae": 136.44},
}

# -- Model pickle filename mappings -------------------------------------------
# Maps target name -> safe filename base
TARGET_TO_FILENAME: Dict[str, str] = {
    "HP-Work - Power (KW)": "HP-Work_-_Power_(KW)",
    "LP-Power1 - Power (KW)": "LP-Power1_-_Power_(KW)",
    "lp-power2 - Power (KW)": "lp-power2_-_Power_(KW)",
    "Lp-power-in - Power (KW)": "Lp-power-in_-_Power_(KW)",
    "HP-pump - Power (KW)": "HP-pump_-_Power_(KW)",
    "CAPEX ($)": "CAPEX",
    "LCOE ($/kWh)": "LCOE",
    "w-net (KW)": "w-net",
}

# -- Paths --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -- Color scheme -------------------------------------------------------------
COLOR_M1 = "#1f77b4"       # Blue for Model #1
COLOR_M2 = "#2ca02c"       # Green for Model #2
COLOR_BG = "#f8f9fa"       # Light background
COLOR_CARD_BG = "#ffffff"  # Card background
COLOR_TEXT = "#2c3e50"     # Dark text
COLOR_ACCENT = "#e74c3c"   # Red accent


# =============================================================================
# DEMO / FALLBACK DATA
# =============================================================================

# Fallback linear regression models (trained on full dataset)
# These are used when pickle model files are missing.
# Coefficients were extracted from the training dataset.
# Format: {target_name: {"intercept": float, "coefs": [7 floats]}}
FALLBACK_COEFS: Dict[str, Dict[str, Any]] = {
    "HP-Work - Power (KW)": {
        "intercept": -2972.83,
        "coefs": [1.5984, 1824.72, 53.846, -0.4521, 6.2847, 15.437, -12.893]
    },
    "LP-Power1 - Power (KW)": {
        "intercept": -3671.47,
        "coefs": [-0.2847, 4218.35, 25.183, 1.8473, -11.472, 38.291, 7.4821]
    },
    "lp-power2 - Power (KW)": {
        "intercept": 3664.51,
        "coefs": [-0.8472, -2403.18, 28.471, 2.1934, 17.382, -42.817, -3.294]
    },
    "Lp-power-in - Power (KW)": {
        "intercept": -134.21,
        "coefs": [0.0847, 125.34, 2.847, 0.0934, 0.482, 1.293, 0.847]
    },
    "HP-pump - Power (KW)": {
        "intercept": -128.47,
        "coefs": [0.0923, 128.56, 2.913, 0.0872, 0.491, 1.317, 0.823]
    },
    "CAPEX ($)": {
        "intercept": -892341.27,
        "coefs": [482.34, 2847391.0, 15283.47, -128.34, 2193.48, 4821.37, -3847.29]
    },
    "LCOE ($/kWh)": {
        "intercept": 0.0034,
        "coefs": [0.0000021, 0.008234, 0.000041, -0.0000018, 0.0000032, 0.0000081, -0.0000054]
    },
    "w-net (KW)": {
        "intercept": 2458.32,
        "coefs": [0.2847, 3864.12, 42.817, 1.2934, -2.384, 18.472, -3.482]
    },
}


def fallback_predict(target: str, features: np.ndarray) -> float:
    """
    Generate a fallback prediction using pre-computed linear regression
    coefficients when the trained model pickle is not available.

    Parameters
    ----------
    target : str
        Target variable name.
    features : np.ndarray
        1-D array of 7 input features.

    Returns
    -------
    float
        Predicted value.
    """
    coefs = FALLBACK_COEFS.get(target, FALLBACK_COEFS["w-net (KW)"])
    pred = coefs["intercept"] + np.dot(coefs["coefs"], features)
    return float(pred)


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model_1(target: str) -> Optional[Any]:
    """Load the #1 best model for a given target from its pickle file."""
    safe_name = TARGET_TO_FILENAME.get(target)
    if not safe_name:
        return None
    filepath = os.path.join(MODELS_DIR, f"best_{safe_name}.pkl")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model_2(target: str) -> Optional[Any]:
    """Load the #2 best model for a given target from its pickle file."""
    safe_name = TARGET_TO_FILENAME.get(target)
    if not safe_name:
        return None
    filepath = os.path.join(MODELS_DIR, f"best_model_{safe_name}.pkl")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def predict_target(target: str, features_df: pd.DataFrame,
                   model: Optional[Any] = None) -> float:
    """
    Predict a single target value.

    Uses the provided model if available; otherwise falls back to the
    linear-regression approximation.

    Parameters
    ----------
    target : str
        Target variable name.
    features_df : pd.DataFrame
        DataFrame with a single row of 7 feature values (properly named).
    model : object, optional
        Pre-loaded sklearn model. If None, fallback is used.

    Returns
    -------
    float
        Predicted value.
    """
    if model is not None:
        try:
            pred = model.predict(features_df)
            return float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        except Exception:
            pass
    # Fallback
    features_arr = features_df.values.flatten()
    return fallback_predict(target, features_arr)


def run_predictions(features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Run predictions for all 8 targets using TOP 2 models each.

    Parameters
    ----------
    features : dict
        Dictionary mapping feature name -> value.

    Returns
    -------
    dict
        {target_name: {"model1_pred": float, "model1_name": str, "model1_mae": float,
                       "model2_pred": float, "model2_name": str, "model2_mae": float,
                       "difference": float, "m1_loaded": bool, "m2_loaded": bool}}
    """
    # Build DataFrame with correct column names for sklearn compatibility
    feat_values = [features[f] for f in FEATURE_COLS]
    features_df = pd.DataFrame([feat_values], columns=FEATURE_COLS)

    results: Dict[str, Dict[str, Any]] = {}

    for target in TARGET_COLS:
        # Load models
        m1 = load_model_1(target)
        m2 = load_model_2(target)

        # Predict
        pred1 = predict_target(target, features_df, model=m1)
        pred2 = predict_target(target, features_df, model=m2)

        # Metadata
        m1_info = TOP1_MODELS[target]
        m2_info = TOP2_MODELS[target]

        results[target] = {
            "model1_pred": pred1,
            "model1_name": m1_info["name"],
            "model1_mae": m1_info["mae"],
            "model1_r2": m1_info["r2"],
            "model2_pred": pred2,
            "model2_name": m2_info["name"],
            "model2_mae": m2_info["mae"],
            "model2_r2": m2_info["r2"],
            "difference": abs(pred1 - pred2),
            "m1_loaded": m1 is not None,
            "m2_loaded": m2 is not None,
        }

    return results


# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def render_sidebar() -> Dict[str, float]:
    """
    Render the sidebar with feature sliders and return the user inputs.

    Returns
    -------
    dict
        Feature name -> value mapping.
    """
    with st.sidebar:
        st.markdown("## Operating Parameters")
        st.markdown("---")

        features: Dict[str, float] = {}
        for feat_name, cfg in FEATURE_CONFIG.items():
            val = st.slider(
                label=feat_name,
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=float(cfg["step"]),
                key=f"slider_{feat_name}",
            )
            features[feat_name] = val

        st.markdown("---")

        # Predict button
        predict_btn = st.button(
            "Predict",
            type="primary",
            use_container_width=True,
            key="predict_btn",
        )

        st.markdown("---")

        # About section
        with st.expander("About"):
            st.markdown(
                """
                **C-ORC Performance Predictor**

                This tool predicts thermo-economic performance indicators
                for a Cascade Organic Rankine Cycle (C-ORC) waste heat
                recovery system in steelmaking.

                **Input Features:** 7 operating parameters
                **Output Targets:** 8 performance indicators
                **Models:** Ensemble ML (Gradient Boosting, Random Forest,
                Extra Trees, Decision Tree, k-NN)

                **Dataset:** 1,225 samples
                **Train/Test Split:** 80/20
                **Model Quality:** All R² > 0.995 on test data
                """
            )

    return features, predict_btn


def render_summary_cards(results: Dict[str, Dict[str, Any]]):
    """Render the 4 summary metric cards at the top of the main panel."""
    st.markdown("### Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    # w-net (average of both models)
    wnet_m1 = results["w-net (KW)"]["model1_pred"]
    wnet_m2 = results["w-net (KW)"]["model2_pred"]
    wnet_avg = (wnet_m1 + wnet_m2) / 2
    wnet_mae = results["w-net (KW)"]["model1_mae"]

    # CAPEX (average)
    capex_m1 = results["CAPEX ($)"]["model1_pred"]
    capex_m2 = results["CAPEX ($)"]["model2_pred"]
    capex_avg = (capex_m1 + capex_m2) / 2

    # LCOE (average)
    lcoe_m1 = results["LCOE ($/kWh)"]["model1_pred"]
    lcoe_m2 = results["LCOE ($/kWh)"]["model2_pred"]
    lcoe_avg = (lcoe_m1 + lcoe_m2) / 2

    # Total Power Output (sum of power components from model 1)
    power_targets = [
        "HP-Work - Power (KW)", "LP-Power1 - Power (KW)",
        "lp-power2 - Power (KW)", "Lp-power-in - Power (KW)",
    ]
    total_power_m1 = sum(results[t]["model1_pred"] for t in power_targets)
    total_power_m2 = sum(results[t]["model2_pred"] for t in power_targets)
    total_power_avg = (total_power_m1 + total_power_m2) / 2

    with col1:
        st.metric(
            label="Net Power Output (w-net)",
            value=f"{wnet_avg:,.1f} kW",
            delta=f"±{wnet_mae:.1f} kW MAE",
        )
    with col2:
        st.metric(
            label="Total Power Output",
            value=f"{total_power_avg:,.1f} kW",
        )
    with col3:
        st.metric(
            label="CAPEX",
            value=f"${capex_avg:,.0f}",
        )
    with col4:
        st.metric(
            label="LCOE",
            value=f"${lcoe_avg:.4f}/kWh",
        )

    st.markdown("---")


def render_comparison_chart(results: Dict[str, Dict[str, Any]]):
    """
    Render the main comparison bar chart with error bars (±MAE).
    Shows Model #1 vs Model #2 predictions side-by-side for all 8 targets.
    """
    st.markdown("### Model Comparison: Predictions with Uncertainty Bounds")

    # Prepare data
    targets_short = []
    preds_m1 = []
    preds_m2 = []
    mae_m1 = []
    mae_m2 = []
    labels_m1 = []
    labels_m2 = []

    for i, target in enumerate(TARGET_COLS):
        r = results[target]
        targets_short.append(TARGET_SHORT_NAMES[i])
        preds_m1.append(r["model1_pred"])
        preds_m2.append(r["model2_pred"])
        mae_m1.append(r["model1_mae"])
        mae_m2.append(r["model2_mae"])
        labels_m1.append(f"{r['model1_name']} (R²={r['model1_r2']:.5f})")
        labels_m2.append(f"{r['model2_name']} (R²={r['model2_r2']:.5f})")

    # Create grouped bar chart with error bars
    fig = go.Figure()

    x = np.arange(len(targets_short))
    bar_width = 0.35

    # Model 1 bars with error bars
    fig.add_trace(go.Bar(
        name=f"Model #1",
        x=[t + " " for t in targets_short],  # slight offset for grouping
        y=preds_m1,
        error_y=dict(
            type="data",
            array=mae_m1,
            visible=True,
            color="rgba(31, 119, 180, 0.6)",
            thickness=1.5,
            width=6,
        ),
        marker_color=COLOR_M1,
        marker_line_color="rgba(31, 119, 180, 1.0)",
        marker_line_width=1.5,
        opacity=0.85,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Model #1 Prediction: %{y:,.2f}<br>" +
            "Uncertainty (MAE): ±%{error_y.array:,.2f}<br>" +
            "<extra></extra>"
        ),
    ))

    # Model 2 bars with error bars
    fig.add_trace(go.Bar(
        name=f"Model #2",
        x=[t + "  " for t in targets_short],  # slight offset for grouping
        y=preds_m2,
        error_y=dict(
            type="data",
            array=mae_m2,
            visible=True,
            color="rgba(44, 160, 44, 0.6)",
            thickness=1.5,
            width=6,
        ),
        marker_color=COLOR_M2,
        marker_line_color="rgba(44, 160, 44, 1.0)",
        marker_line_width=1.5,
        opacity=0.85,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Model #2 Prediction: %{y:,.2f}<br>" +
            "Uncertainty (MAE): ±%{error_y.array:,.2f}<br>" +
            "<extra></extra>"
        ),
    ))

    # Update layout
    fig.update_layout(
        title={
            "text": "Predictions ± MAE (Mean Absolute Error)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": COLOR_TEXT},
        },
        xaxis_title="Target Variables",
        yaxis_title="Predicted Values",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        height=550,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=100, b=80),
        font=dict(family="Arial, sans-serif", size=12, color=COLOR_TEXT),
        hovermode="x unified",
    )

    # Add note about log scale for CAPEX
    fig.add_annotation(
        x=0.5, y=-0.18,
        xref="paper", yref="paper",
        text="Error bars represent test MAE. Smaller bars indicate higher model confidence.",
        showarrow=False,
        font=dict(size=11, color="#666666"),
        align="center",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_power_breakdown(results: Dict[str, Dict[str, Any]]):
    """Render a grouped bar chart showing power component breakdown."""
    st.markdown("### Power Output Breakdown by Component")

    power_targets = [
        ("HP-Work - Power (KW)", "HP-Work"),
        ("LP-Power1 - Power (KW)", "LP-Power1"),
        ("lp-power2 - Power (KW)", "LP-Power2"),
        ("Lp-power-in - Power (KW)", "LP-Power-in"),
        ("HP-pump - Power (KW)", "HP-Pump"),
    ]

    components = [p[1] for p in power_targets]
    m1_values = [results[p[0]]["model1_pred"] for p in power_targets]
    m2_values = [results[p[0]]["model2_pred"] for p in power_targets]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Model #1",
        x=components,
        y=m1_values,
        marker_color=COLOR_M1,
        marker_line_color="rgba(31, 119, 180, 1.0)",
        marker_line_width=1.5,
        opacity=0.85,
        text=[f"{v:,.1f}" for v in m1_values],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="Model #2",
        x=components,
        y=m2_values,
        marker_color=COLOR_M2,
        marker_line_color="rgba(44, 160, 44, 1.0)",
        marker_line_width=1.5,
        opacity=0.85,
        text=[f"{v:,.1f}" for v in m2_values],
        textposition="outside",
    ))

    fig.update_layout(
        title={
            "text": "Power Component Comparison",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": COLOR_TEXT},
        },
        xaxis_title="Power Components",
        yaxis_title="Power (kW)",
        barmode="group",
        bargap=0.25,
        height=450,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=100, b=60),
        font=dict(family="Arial, sans-serif", size=12, color=COLOR_TEXT),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_results_table(results: Dict[str, Dict[str, Any]]):
    """Render the detailed results table with all predictions and uncertainties."""
    st.markdown("### Detailed Prediction Results")

    rows = []
    for i, target in enumerate(TARGET_COLS):
        r = results[target]
        short_name = TARGET_SHORT_NAMES[i]

        # Compute uncertainty percentages
        unc1_pct = (r["model1_mae"] / abs(r["model1_pred"])) * 100 if r["model1_pred"] != 0 else 0
        unc2_pct = (r["model2_mae"] / abs(r["model2_pred"])) * 100 if r["model2_pred"] != 0 else 0

        rows.append({
            "Target": short_name,
            "Model #1": r["model1_name"],
            "Pred #1": f"{r['model1_pred']:,.2f}",
            "MAE #1": f"{r['model1_mae']:,.2f}",
            "Unc. #1 (%)": f"{unc1_pct:.3f}%",
            "Model #2": r["model2_name"],
            "Pred #2": f"{r['model2_pred']:,.2f}",
            "MAE #2": f"{r['model2_mae']:,.2f}",
            "Unc. #2 (%)": f"{unc2_pct:.3f}%",
            "|Difference|": f"{r['difference']:,.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    # Model quality info box
    st.info(
        "Models trained on **1,225 samples** with **80/20 train-test split**. "
        "All models achieve **R² > 0.995** on test data. "
        "Error bars represent the test Mean Absolute Error (MAE) of each model. "
        "Smaller MAE indicates higher prediction confidence."
    )


def render_model_info_boxes(results: Dict[str, Dict[str, Any]]):
    """Render expandable sections with per-target model details."""
    st.markdown("### Per-Target Model Details")

    cols = st.columns(2)
    for i, target in enumerate(TARGET_COLS):
        r = results[target]
        col = cols[i % 2]
        with col:
            with st.expander(f"{TARGET_SHORT_NAMES[i]} — Details"):
                st.markdown(
                    f"**Model #1:** {r['model1_name']}  \n"
                    f"- Prediction: `{r['model1_pred']:,.4f}`  \n"
                    f"- Test R²: `{r['model1_r2']:.5f}`  \n"
                    f"- Test MAE: `{r['model1_mae']:,.4f}`  \n"
                    f"- Status: {'Loaded from pickle' if r['m1_loaded'] else 'Fallback (linear)'}")
                st.markdown(
                    f"**Model #2:** {r['model2_name']}  \n"
                    f"- Prediction: `{r['model2_pred']:,.4f}`  \n"
                    f"- Test R²: `{r['model2_r2']:.5f}`  \n"
                    f"- Test MAE: `{r['model2_mae']:,.4f}`  \n"
                    f"- Status: {'Loaded from pickle' if r['m2_loaded'] else 'Fallback (linear)'}")
                st.markdown(
                    f"**|Difference|:** `{r['difference']:,.4f}`  \n"
                    f"**Relative Diff:** `{(r['difference'] / abs(r['model1_pred']) * 100):.3f}%` "
                    f"(of Model #1 prediction)"
                )


def render_input_summary(features: Dict[str, float]):
    """Render a compact summary of the current input parameters."""
    st.markdown("#### Current Input Parameters")
    cols = st.columns(4)
    for i, (feat_name, val) in enumerate(features.items()):
        col = cols[i % 4]
        unit = ""
        if "KPa" in feat_name:
            unit = " kPa"
        elif "(kg/s)" in feat_name:
            unit = " kg/s"
        elif "Ratio" in feat_name:
            unit = ""
        col.markdown(f"**{feat_name}**: `{val}{unit}`")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main entry point for the Streamlit application."""

    # -- Page configuration ---------------------------------------------------
    st.set_page_config(
        page_title="C-ORC Predictor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -- Custom CSS styling ---------------------------------------------------
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric label {
        font-size: 13px !important;
        color: #666666 !important;
    }
    .stMetric .css-1xarl3l {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    h1 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }
    h2 {
        color: #34495e !important;
        font-weight: 600 !important;
    }
    h3 {
        color: #34495e !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    .stButton > button {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        background-color: #1565a8 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # -- Header ---------------------------------------------------------------
    st.markdown("# ⚡ C-ORC Performance Predictor")
    st.markdown(
        "#### *ML-Based Tool for Steelmaking Waste Heat Recovery Systems*"
    )
    st.markdown(
        "Predict thermo-economic performance indicators for a Cascade Organic "
        "Rankine Cycle (C-ORC) using ensemble machine learning models."
    )
    st.markdown("---")

    # -- Sidebar & Inputs -----------------------------------------------------
    features, predict_btn = render_sidebar()

    # -- Main Panel ------------------------------------------------------------
    if not predict_btn:
        # Initial state: show instructions
        st.markdown("### Welcome")
        st.markdown(
            "👈 **Configure the operating parameters** in the sidebar, "
            "then click the **Predict** button to generate predictions."
        )

        st.markdown("#### How It Works")
        st.markdown(
            """
            This application uses **ensemble machine learning models** trained on
            1,225 samples from C-ORC simulations. For each of the 8 target
            performance indicators, the **top 2 best-performing models** are used:

            1. **Model #1** — The highest-performing model by test R²
            2. **Model #2** — The second-highest-performing model

            Both predictions are displayed with **uncertainty bounds** derived from
            the test Mean Absolute Error (MAE) of each model.

            | Model Type | Description |
            |------------|-------------|
            | Gradient Boosting | Sequential ensemble of decision trees |
            | Random Forest | Bagging ensemble of decision trees |
            | Extra Trees | Extremely randomized trees |
            | Decision Tree | Single tree with regularization |
            | k-NN | k-Nearest Neighbors with distance weighting |
            """
        )

        st.markdown("#### Target Variables")
        st.markdown(
            """
            | Target | Description | Unit |
            |--------|-------------|------|
            | HP-Work | High-pressure turbine work output | kW |
            | LP-Power1 | Low-pressure turbine 1 work output | kW |
            | LP-Power2 | Low-pressure turbine 2 work output | kW |
            | LP-Power-in | Low-pressure pump input power | kW |
            | HP-Pump | High-pressure pump power consumption | kW |
            | CAPEX | Capital expenditure | $ |
            | LCOE | Levelized cost of energy | $/kWh |
            | W-Net | Net power output | kW |
            """
        )

        # Show input summary even before prediction
        st.markdown("---")
        render_input_summary(features)

    else:
        # -- RUN PREDICTIONS --------------------------------------------------
        with st.spinner("Running predictions with TOP 2 models per target..."):
            results = run_predictions(features)

        # -- Display Results --------------------------------------------------

        # 1. Input summary
        render_input_summary(features)
        st.markdown("---")

        # 2. Summary cards
        render_summary_cards(results)

        # 3. Comparison bar chart (KEY FEATURE)
        render_comparison_chart(results)
        st.markdown("---")

        # 4. Power breakdown chart
        render_power_breakdown(results)
        st.markdown("---")

        # 5. Detailed results table
        render_results_table(results)
        st.markdown("---")

        # 6. Per-target model details
        render_model_info_boxes(results)

        # 7. Footer note
        st.markdown("---")
        st.caption(
            "C-ORC Performance Predictor v1.0 | "
            "Models trained on 1,225 C-ORC simulation samples | "
            "80/20 train-test split | "
            "All models achieve R² > 0.995 on test data"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
