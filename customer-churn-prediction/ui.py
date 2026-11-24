"""Customer Churn Predictor: Streamlit app for the Telco churn project.

Run from this folder:
    streamlit run ui.py

Needs (next to this file):
    churn_model.pkl, scaler.pkl, threshold.pkl   <- from customer_churn.py
    Telco-Customer-Churn.csv                      <- "Data insights" tab
    model_metrics.csv                             <- "Model performance" tab
                                                     (created by python customer_churn.py)
"""

import base64
import html
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration (keep in sync with customer_churn.py)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "threshold.pkl"
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"
METRICS_PATH = BASE_DIR / "model_metrics.csv"
ARTIFACT_PATHS = (MODEL_PATH, SCALER_PATH, THRESHOLD_PATH)

BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
ONE_HOT_COLS = ["InternetService", "Contract", "PaymentMethod"]
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

YES_NO = ["No", "Yes"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": YES_NO,
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": YES_NO,
    "OnlineBackup": YES_NO,
    "DeviceProtection": YES_NO,
    "TechSupport": YES_NO,
    "StreamingTV": YES_NO,
    "StreamingMovies": YES_NO,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
}
PRESETS = {
    "default": dict(DEFAULTS),
    "churn": {**DEFAULTS, "tenure": 3, "MonthlyCharges": 85.0},
    "stay": {
        **DEFAULTS,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.0,
    },
}

TABS = ["Churn prediction", "Data insights", "Model performance"]

NAVY = "#1f3a5f"
RED, AMBER, GREEN, BLUE, GREY = "#d64545", "#d99a1c", "#2e9d6a", "#3b6ea5", "#7f7f7f"
LEVEL_COLOR = {"High": RED, "Medium": AMBER, "Low": GREEN}
LEVEL_ACTION = {
    "High": "Prioritise for retention outreach.",
    "Medium": "Monitor, and consider a proactive check-in.",
    "Low": "No action needed.",
}

TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-36 mo", "37-48 mo", "49-60 mo", "61-72 mo"]
BREAKDOWN_COLS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "gender",
    "PhoneService",
    "MultipleLines",
    "OnlineBackup",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
]

# --------------------------------------------------------------------------- #
# Icons and favicon (no emoji: line icons are drawn as inline SVG)
# --------------------------------------------------------------------------- #
TREND_DOWN = '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>'
TREND_UP = '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>'
ICON_ALERT = (
    '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
    '<path d="M12 9v4"/><path d="M12 17h.01"/>'
)
ICON_EYE = '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12"/><circle cx="12" cy="12" r="3"/>'
ICON_CHECK = '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>'
LEVEL_ICON = {"High": ICON_ALERT, "Medium": ICON_EYE, "Low": ICON_CHECK}


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode(
        "ascii"
    )


def icon(paths: str, color: str, size: int = 16) -> str:
    """Inline line icon (24px grid) as an <img> so it renders anywhere markdown does."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )
    return (
        f'<img class="ico" src="{_svg_uri(svg)}" width="{size}" height="{size}" alt="">'
    )


LOGO_URI = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    f'<rect width="24" height="24" rx="5.5" fill="{NAVY}"/>'
    '<g transform="translate(3.6 3.6) scale(0.7)" fill="none" stroke="#ffffff" '
    f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{TREND_DOWN}</g></svg>'
)


def make_favicon():
    """Browser-tab icon: white downward trend line on a navy rounded square."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # Pillow ships with Streamlit, but stay safe
        return ":material/trending_down:"

    u = 16  # pixels per grid unit while drawing, then downsampled for smooth edges

    def t(x: float, y: float) -> tuple[float, float]:
        return (3.6 + 0.7 * x) * u, (3.6 + 0.7 * y) * u

    img = Image.new("RGBA", (24 * u, 24 * u), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, 24 * u - 1, 24 * u - 1], radius=int(5.5 * u), fill=NAVY
    )
    width = round(2.2 * 0.7 * u)
    for line in (
        [(22, 17), (13.5, 8.5), (8.5, 13.5), (2, 7)],
        [(16, 17), (22, 17), (22, 11)],
    ):
        pts = [t(x, y) for x, y in line]
        draw.line(pts, fill="white", width=width, joint="curve")
        for px, py in pts:  # round caps and joins
            draw.ellipse(
                [px - width / 2, py - width / 2, px + width / 2, py + width / 2],
                fill="white",
            )
    return img.resize((96, 96), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
/* No sidebar anywhere */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stSidebarNav"] {display: none !important;}
footer, #MainMenu {visibility: hidden;}

.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;
}
.block-container h3, [data-testid="stMainBlockContainer"] h3 {
    font-size: 1.02rem; font-weight: 600; letter-spacing: -0.005em;
}

/* Top bar */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px 24px; margin-bottom: 0.6rem;
}
.brand {display: flex; align-items: center; gap: 12px;}
.brand img {width: 40px; height: 40px; border-radius: 10px; margin: 0;}
.brand-title {font-size: 1.35rem; font-weight: 700; letter-spacing: -0.015em; line-height: 1.2;}
.brand-sub {font-size: 0.86rem; opacity: 0.65; line-height: 1.3;}
.pill {
    display: inline-block; margin: 0 0 0 8px; padding: 3px 12px; font-size: 0.8rem;
    border-radius: 999px; border: 1px solid rgba(128,128,128,0.35); opacity: 0.85;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid rgba(128,128,128,0.25);}
.stTabs [data-baseweb="tab"] {padding: 10px 16px; height: auto;}
.stTabs button[data-baseweb="tab"] p {font-size: 0.97rem; font-weight: 600;}
.intro {margin: 0.4rem 0 1rem 0; font-size: 0.93rem; opacity: 0.7;}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.22);
    border-radius: 10px; padding: 14px 18px;
}
div[data-testid="stMetricLabel"] p {font-size: 0.84rem; opacity: 0.75;}
div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}

/* Input panel */
.panel-title {font-size: 1.05rem; font-weight: 700; margin-bottom: 0.4rem;}
.group {
    font-size: 0.9rem; font-weight: 600; margin: 1.1rem 0 0.5rem 0; padding-bottom: 5px;
    border-bottom: 1px solid rgba(128,128,128,0.25);
}

/* Result card */
.ico {vertical-align: -3px; margin: 0; display: inline-block;}
.risk-card {
    display: flex; flex-wrap: wrap; align-items: center; gap: 12px 28px;
    border: 1px solid rgba(128,128,128,0.22); border-left: 5px solid var(--c);
    border-radius: 10px; padding: 18px 24px; background: rgba(128,128,128,0.06);
}
.risk-label {font-size: 0.86rem; opacity: 0.7;}
.risk-value {
    font-size: 3rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.risk-side {flex: 1 1 240px;}
.badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px 3px 9px;
    font-size: 0.86rem; font-weight: 600; border-radius: 999px; border: 1px solid;
}
.risk-note {margin: 10px 0 0 0; font-size: 0.93rem; opacity: 0.85; line-height: 1.45;}
.factor-row {display: flex; align-items: baseline; gap: 8px; font-size: 0.93rem; line-height: 1.6;}
.factor-key {font-weight: 600; min-width: 92px;}
.disclaimer {font-size: 0.78rem; opacity: 0.6; line-height: 1.45; margin-top: 1.5rem;}
</style>
"""


# --------------------------------------------------------------------------- #
# Loading (cached; stamps refresh the cache when files are re-generated)
# --------------------------------------------------------------------------- #
def _stamp(*paths: Path) -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


@st.cache_resource(show_spinner=False)
def load_artifacts(stamp: tuple) -> dict:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    features = getattr(scaler, "feature_names_in_", None)
    if features is None:
        raise ValueError(
            "scaler.pkl has no feature names. Re-run customer_churn.py so the "
            "scaler is fit on a DataFrame."
        )
    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "features": list(features),
    }


@st.cache_data(show_spinner=False)
def load_dataset(stamp: tuple) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    ).astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_metrics(stamp: tuple) -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def pretty(name: str) -> str:
    """'InternetService_Fiber optic' -> 'Internet Service: Fiber optic'."""
    head, _, tail = name.partition("_")
    head = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", head)
    head = head[:1].upper() + head[1:]
    return f"{head}: {tail}" if tail else head


def group_of(feature: str) -> str:
    """Original input a model feature came from (one-hot columns map back)."""
    for col in ONE_HOT_COLS:
        if feature.startswith(col + "_"):
            return col
    return feature


def display_value(var: str, raw: dict) -> str:
    value = raw[var]
    if var == "tenure":
        return f"{value} mo"
    if var == "MonthlyCharges":
        return f"${value:,.0f}"
    if var == "SeniorCitizen":
        return "Yes" if value == 1 else "No"
    return str(value)


def read_inputs() -> dict:
    """Current form values as a raw customer, with dependent fields made consistent."""
    raw = {k: st.session_state[k] for k in DEFAULTS}
    raw["SeniorCitizen"] = 1 if raw["SeniorCitizen"] == "Yes" else 0
    if raw["PhoneService"] == "No":
        raw["MultipleLines"] = "No"
    if raw["InternetService"] == "No":
        for col in INTERNET_ADDONS:
            raw[col] = "No"
    return raw


def encode_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Same encoding as training. get_dummies without drop_first + reindex drops
    the baseline levels (DSL, Month-to-month, Bank transfer) automatically."""
    X = df.copy()
    for col in BINARY_COLS:
        X[col] = X[col].map(BINARY_MAP)
    X = pd.get_dummies(X, columns=ONE_HOT_COLS, dtype=int)
    return X.reindex(columns=features, fill_value=0)


def score(raw: dict, art: dict) -> tuple[float, np.ndarray]:
    """Returns (churn probability, per-feature contribution in log-odds)."""
    scaled = art["scaler"].transform(encode_frame(pd.DataFrame([raw]), art["features"]))
    proba = float(art["model"].predict_proba(scaled)[0, 1])
    return proba, art["model"].coef_[0] * scaled[0]


def risk_level(p: float, threshold: float) -> str:
    if p >= threshold:
        return "High"
    return "Medium" if p >= threshold / 2 else "Low"


def variable_effects(
    contrib: np.ndarray, features: list[str], raw: dict
) -> pd.DataFrame:
    """Contribution per customer attribute (one-hot columns summed back together)."""
    effects = pd.Series(contrib, index=features).groupby(group_of).sum()
    out = pd.DataFrame(
        {
            "label": [f"{pretty(v)}: {display_value(v, raw)}" for v in effects.index],
            "effect": effects.to_numpy(),
        }
    )
    return out.reindex(
        out["effect"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def show_chart(chart: alt.Chart) -> None:
    chart = (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            gridOpacity=0.25,
        )
        .configure_legend(labelFontSize=12, titleFontSize=12)
    )
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def show_table(df: pd.DataFrame, column_config: dict | None = None) -> None:
    try:
        st.dataframe(df, hide_index=True, column_config=column_config, width="stretch")
    except TypeError:
        st.dataframe(
            df, hide_index=True, column_config=column_config, use_container_width=True
        )


def stretch_button(label: str, **kwargs) -> bool:
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def top_bar(art: dict) -> None:
    pills = "".join(
        f'<span class="pill">{html.escape(t)}</span>'
        for t in ("Logistic regression", f"{len(art['features'])} features")
    )
    st.markdown(
        '<div class="topbar">'
        f'<div class="brand"><img src="{LOGO_URI}" alt="">'
        '<div><div class="brand-title">Customer Churn Predictor</div>'
        '<div class="brand-sub">Telco customer retention analytics</div></div></div>'
        f"<div>{pills}</div></div>",
        unsafe_allow_html=True,
    )


def intro(text: str) -> None:
    st.markdown(f'<div class="intro">{html.escape(text)}</div>', unsafe_allow_html=True)


def group(title: str) -> None:
    st.markdown(
        f'<div class="group">{html.escape(title)}</div>', unsafe_allow_html=True
    )


def apply_preset(name: str) -> None:
    st.session_state.update(PRESETS[name])


def select_row(*specs: tuple) -> None:
    """One row of selectboxes. Each spec is (label, key) or (label, key, disabled)."""
    for col, (label, key, *rest) in zip(st.columns(len(specs)), specs):
        with col:
            st.selectbox(label, OPTIONS[key], key=key, disabled=bool(rest and rest[0]))


def risk_card(p: float, level: str, threshold: float) -> str:
    color = LEVEL_COLOR[level]
    side = "above" if p >= threshold else "below"
    return (
        f'<div class="risk-card" style="--c:{color}">'
        '<div><div class="risk-label">Churn probability</div>'
        f'<div class="risk-value">{p:.1%}</div></div>'
        '<div class="risk-side">'
        f'<span class="badge" style="color:{color};background:{color}22;border-color:{color}66">'
        f"{icon(LEVEL_ICON[level], color)}{level} risk</span>"
        f'<p class="risk-note">This is {side} the {threshold:.0%} decision threshold. '
        f"{LEVEL_ACTION[level]}</p></div></div>"
    )


def risk_scale(p: float, threshold: float) -> alt.Chart:
    """Low / Medium / High zones with a marker at the customer's probability."""
    x_scale = alt.Scale(domain=[0, 1])
    zones = pd.DataFrame(
        {
            "zone": ["Low", "Medium", "High"],
            "lo": [0.0, threshold / 2, threshold],
            "hi": [threshold / 2, threshold, 1.0],
        }
    )
    bands = (
        alt.Chart(zones)
        .mark_bar(height=26, opacity=0.45)
        .encode(
            x=alt.X(
                "lo:Q",
                scale=x_scale,
                axis=alt.Axis(format="%", title="Churn probability"),
            ),
            x2="hi:Q",
            color=alt.Color(
                "zone:N",
                title=None,
                scale=alt.Scale(
                    domain=list(LEVEL_COLOR)[::-1],
                    range=[LEVEL_COLOR[k] for k in list(LEVEL_COLOR)[::-1]],
                ),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
        )
    )
    point = pd.DataFrame({"p": [p], "label": [f"{p:.1%}"]})
    color = LEVEL_COLOR[risk_level(p, threshold)]
    marker = (
        alt.Chart(point)
        .mark_rule(strokeWidth=5, color=color)
        .encode(x=alt.X("p:Q", scale=x_scale))
    )
    label = (
        alt.Chart(point)
        .mark_text(fontSize=14, fontWeight="bold", dy=-2)
        .encode(x=alt.X("p:Q", scale=x_scale), y=alt.value(10), text="label:N")
    )
    return alt.layer(bands, marker, label).properties(height=110)


def factor_lines(effects: pd.DataFrame) -> str:
    def fmt(labels: list[str]) -> str:
        return (
            html.escape(", ".join(labels)).replace("$", "&#36;") if labels else "None"
        )

    up = (
        effects[effects["effect"] > 0.05]
        .sort_values("effect", ascending=False)["label"]
        .head(3)
    )
    down = effects[effects["effect"] < -0.05].sort_values("effect")["label"].head(3)
    return (
        f'<div class="factor-row">{icon(TREND_UP, RED)}'
        f'<span class="factor-key" style="color:{RED}">Raises risk</span>'
        f"<span>{fmt(list(up))}</span></div>"
        f'<div class="factor-row">{icon(TREND_DOWN, GREEN)}'
        f'<span class="factor-key" style="color:{GREEN}">Lowers risk</span>'
        f"<span>{fmt(list(down))}</span></div>"
    )


# --------------------------------------------------------------------------- #
# Tab: Churn prediction
# --------------------------------------------------------------------------- #
def input_panel(art: dict) -> float:
    """Customer form (left column). Returns the decision threshold."""
    ss = st.session_state
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Customer profile</div>', unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            stretch_button(
                "High-risk example",
                on_click=apply_preset,
                args=("churn",),
                key="btn_churn",
            )
        with c2:
            stretch_button(
                "Loyal example", on_click=apply_preset, args=("stay",), key="btn_stay"
            )
        with c3:
            stretch_button(
                "Reset", on_click=apply_preset, args=("default",), key="btn_reset"
            )

        group("Account holder")
        select_row(
            ("Gender", "gender"),
            ("Senior citizen", "SeniorCitizen"),
            ("Partner", "Partner"),
        )
        a, b = st.columns([1, 2])
        with a:
            st.selectbox("Dependents", OPTIONS["Dependents"], key="Dependents")
        with b:
            st.slider("Tenure (months)", 0, 72, key="tenure")

        group("Phone and internet")
        no_phone = ss["PhoneService"] == "No"
        no_net = ss["InternetService"] == "No"
        select_row(
            ("Phone service", "PhoneService"),
            ("Multiple lines", "MultipleLines", no_phone),
            ("Internet service", "InternetService"),
        )
        select_row(
            ("Online security", "OnlineSecurity", no_net),
            ("Online backup", "OnlineBackup", no_net),
            ("Device protection", "DeviceProtection", no_net),
        )
        select_row(
            ("Tech support", "TechSupport", no_net),
            ("Streaming TV", "StreamingTV", no_net),
            ("Streaming movies", "StreamingMovies", no_net),
        )
        if no_net or no_phone:
            st.caption("Fields that need phone or internet service are treated as No.")

        group("Contract and billing")
        select_row(("Contract", "Contract"), ("Payment method", "PaymentMethod"))
        a, b = st.columns(2)
        with a:
            st.selectbox(
                "Paperless billing", OPTIONS["PaperlessBilling"], key="PaperlessBilling"
            )
        with b:
            st.number_input(
                "Monthly charges ($)", 18.0, 120.0, step=0.5, key="MonthlyCharges"
            )

        group("Decision threshold")
        threshold = st.slider(
            "Flag customers at or above this probability",
            0.05,
            0.95,
            float(art["threshold"]),
            0.01,
            key="decision_threshold",
            help="Lower it to catch more churners at the cost of more false alarms.",
        )
        st.caption(f"Saved threshold from training: {art['threshold']:.2f}")
    return threshold


def page_prediction(art: dict) -> None:
    intro(
        "Score a single customer, see what drives the risk, and test what-if changes."
    )
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        threshold = input_panel(art)

    raw = read_inputs()
    p, contrib = score(raw, art)
    level = risk_level(p, threshold)
    effects = variable_effects(contrib, art["features"], raw)

    with right:
        st.markdown(risk_card(p, level, threshold), unsafe_allow_html=True)
        st.write("")
        show_chart(risk_scale(p, threshold))
        st.markdown(factor_lines(effects), unsafe_allow_html=True)

        st.write("")
        st.subheader("What drives this score")
        top = effects.head(8)
        chart = (
            alt.Chart(top)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("effect:Q", title="Effect on churn risk (log-odds)"),
                y=alt.Y("label:N", sort=top["label"].tolist(), title=None),
                color=alt.condition(
                    alt.datum.effect > 0, alt.value(RED), alt.value(GREEN)
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Customer attribute"),
                    alt.Tooltip("effect:Q", title="Effect", format="+.2f"),
                ],
            )
            .properties(height=280)
        )
        show_chart(chart)
        st.caption(
            "Compared with an average customer: red raises churn risk, green lowers it."
        )

        st.subheader("What-if scenarios")
        render_scenarios(raw, p, art, threshold)


def render_scenarios(raw: dict, p: float, art: dict, threshold: float) -> None:
    options = [("Current profile", {})]
    for contract in ("One year", "Two year"):
        if raw["Contract"] != contract:
            options.append((f"Contract: {contract}", {"Contract": contract}))
    if raw["InternetService"] != "No":
        if raw["TechSupport"] == "No":
            options.append(("Add tech support", {"TechSupport": "Yes"}))
        if raw["OnlineSecurity"] == "No":
            options.append(("Add online security", {"OnlineSecurity": "Yes"}))
    if raw["PaymentMethod"] in ("Electronic check", "Mailed check"):
        options.append(
            ("Pay by card (automatic)", {"PaymentMethod": "Credit card (automatic)"})
        )

    sc = pd.DataFrame(
        [
            {
                "scenario": label,
                "probability": p if not change else score({**raw, **change}, art)[0],
            }
            for label, change in options
        ]
    )
    sc["change_pp"] = (sc["probability"] - p) * 100

    x = alt.X(
        "probability:Q",
        title="Churn probability",
        axis=alt.Axis(format="%"),
        scale=alt.Scale(domain=[0, 1.1]),
    )
    y = alt.Y("scenario:N", sort=sc["scenario"].tolist(), title=None)
    bars = (
        alt.Chart(sc)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=x,
            y=y,
            color=alt.condition(
                alt.datum.probability >= threshold, alt.value(RED), alt.value(BLUE)
            ),
            tooltip=[
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("probability:Q", title="Churn probability", format=".1%"),
                alt.Tooltip("change_pp:Q", title="Change (points)", format="+.1f"),
            ],
        )
    )
    labels = (
        alt.Chart(sc)
        .mark_text(align="left", dx=4)
        .encode(x=x, y=y, text=alt.Text("probability:Q", format=".1%"))
    )
    show_chart((bars + labels).properties(height=44 * len(sc) + 30))
    st.caption(
        "Red bars are at or above the decision threshold. Estimates show associations "
        "in historical data, not guaranteed outcomes."
    )


# --------------------------------------------------------------------------- #
# Tab: Data insights
# --------------------------------------------------------------------------- #
def page_insights() -> None:
    intro("How churn varies across the Telco customer base.")
    if not DATA_PATH.exists():
        st.info(f"Put {DATA_PATH.name} next to ui.py to see data insights.")
        return
    df = load_dataset(_stamp(DATA_PATH))
    overall = df["Churn"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Churn rate", f"{overall:.1%}")
    k3.metric("Average tenure", f"{df['tenure'].mean():.0f} months")
    k4.metric("Average monthly charges", f"${df['MonthlyCharges'].mean():.2f}")
    st.write("")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Churn rate by segment")
        cols = [c for c in BREAKDOWN_COLS if c in df.columns]
        col = st.selectbox("Break down by", cols, format_func=pretty)
        seg = df.groupby(col)["Churn"].agg(rate="mean", customers="size").reset_index()
        bars = (
            alt.Chart(seg)
            .mark_bar(color=BLUE, cornerRadiusEnd=3)
            .encode(
                x=alt.X("rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
                y=alt.Y(f"{col}:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip(f"{col}:N", title=pretty(col)),
                    alt.Tooltip("rate:Q", title="Churn rate", format=".1%"),
                    alt.Tooltip("customers:Q", title="Customers"),
                ],
            )
        )
        avg_line = (
            alt.Chart(pd.DataFrame({"r": [overall]}))
            .mark_rule(color=GREY, strokeDash=[5, 3], strokeWidth=2)
            .encode(x="r:Q")
        )
        show_chart(bars + avg_line)
        worst = seg.sort_values("rate").iloc[-1]
        st.caption(
            f"Dashed line = overall churn rate. Highest in this view: {worst[col]} "
            f"({worst['rate']:.0%}, {int(worst['customers']):,} customers)."
        )
    with right:
        st.subheader("Churn rate by tenure")
        ten = (
            df.groupby("tenure_group")["Churn"]
            .agg(rate="mean", customers="size")
            .reset_index()
        )
        chart = (
            alt.Chart(ten)
            .mark_bar(color=BLUE, cornerRadiusEnd=3)
            .encode(
                x=alt.X(
                    "tenure_group:N",
                    sort=TENURE_LABELS,
                    title="Tenure",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("tenure_group:N", title="Tenure"),
                    alt.Tooltip("rate:Q", title="Churn rate", format=".1%"),
                    alt.Tooltip("customers:Q", title="Customers"),
                ],
            )
            .properties(height=300)
        )
        show_chart(chart)
        st.caption("New customers churn far more often than long-standing ones.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Churn rate by contract and internet service")
        heat = (
            df.groupby(["Contract", "InternetService"])["Churn"]
            .agg(rate="mean", customers="size")
            .reset_index()
        )
        base = alt.Chart(heat).encode(
            x=alt.X(
                "InternetService:N",
                title="Internet service",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("Contract:N", title=None, sort=OPTIONS["Contract"]),
        )
        rect = base.mark_rect().encode(
            color=alt.Color(
                "rate:Q", scale=alt.Scale(scheme="reds", domain=[0, 0.6]), legend=None
            ),
            tooltip=[
                alt.Tooltip("Contract:N"),
                alt.Tooltip("InternetService:N", title="Internet service"),
                alt.Tooltip("rate:Q", title="Churn rate", format=".1%"),
                alt.Tooltip("customers:Q", title="Customers"),
            ],
        )
        text = base.mark_text(fontSize=13, fontWeight="bold").encode(
            text=alt.Text("rate:Q", format=".0%"),
            color=alt.condition(
                alt.datum.rate > 0.3, alt.value("white"), alt.value("black")
            ),
        )
        show_chart((rect + text).properties(height=220))
    with right:
        st.subheader("Monthly charges by outcome")
        d = df.assign(outcome=df["Churn"].map({1: "Churned", 0: "Stayed"}))
        hist = (
            alt.Chart(d)
            .mark_bar(opacity=0.6)
            .encode(
                x=alt.X(
                    "MonthlyCharges:Q",
                    bin=alt.Bin(maxbins=30),
                    title="Monthly charges ($)",
                ),
                y=alt.Y("count()", stack=None, title="Customers"),
                color=alt.Color(
                    "outcome:N",
                    title="Outcome",
                    scale=alt.Scale(domain=["Stayed", "Churned"], range=[BLUE, RED]),
                ),
            )
            .properties(height=220)
        )
        show_chart(hist)


# --------------------------------------------------------------------------- #
# Tab: Model performance
# --------------------------------------------------------------------------- #
def page_performance(art: dict) -> None:
    intro("How the model was evaluated and what it has learned.")
    model = art["model"]

    if METRICS_PATH.exists():
        metrics = load_metrics(_stamp(METRICS_PATH))
        final = metrics[metrics["model"].str.contains("final", case=False)]
        if not final.empty:
            f = final.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Recall",
                f"{f['recall']:.1%}",
                help="Share of actual churners the model catches.",
            )
            m2.metric(
                "Precision",
                f"{f['precision']:.1%}",
                help="Share of flagged customers who actually churn.",
            )
            m3.metric(
                "F1 score", f"{f['f1']:.1%}", help="Balance of precision and recall."
            )
            m4.metric(
                "ROC-AUC",
                f"{f['roc_auc']:.3f}",
                help="How well the model ranks churners above non-churners.",
            )
            st.caption(
                f"Final model on the held-out test set at threshold {f['threshold']:.2f}."
            )
        st.write("")

        left, right = st.columns([2, 1], gap="large")
        with left:
            st.subheader("Model comparison")
            long = metrics.melt(
                id_vars=["model"],
                value_vars=["recall", "precision", "roc_auc"],
                var_name="metric",
                value_name="value",
            )
            chart = (
                alt.Chart(long)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("model:N", title=None, sort=metrics["model"].tolist()),
                    yOffset="metric:N",
                    color=alt.Color("metric:N", title="Metric"),
                    tooltip=[
                        "model:N",
                        "metric:N",
                        alt.Tooltip("value:Q", format=".3f"),
                    ],
                )
                .properties(height=70 * len(metrics) + 40)
            )
            show_chart(chart)
        with right:
            st.subheader("Model card")
            st.markdown(
                "| | |\n|---|---|\n"
                "| Algorithm | Logistic regression |\n"
                f"| Regularisation (C) | {getattr(model, 'C', float('nan')):g} |\n"
                f"| Class weight | {getattr(model, 'class_weight', None)} |\n"
                f"| Features | {len(art['features'])} |\n"
                f"| Saved threshold | {art['threshold']:.2f} |"
            )
        st.subheader("Evaluation table")
        show_table(metrics)
    else:
        st.info(
            f"{METRICS_PATH.name} not found. Run `python customer_churn.py` to "
            "generate the model comparison."
        )

    st.subheader("Feature effects")
    coefs = model.coef_[0]
    eff = pd.DataFrame(
        {
            "feature": [pretty(f) for f in art["features"]],
            "coef": coefs,
            "odds_ratio": np.exp(coefs),
        }
    ).sort_values("coef", ascending=False)
    chart = (
        alt.Chart(eff)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("coef:Q", title="Effect on churn odds (log scale, per 1 SD)"),
            y=alt.Y("feature:N", sort=eff["feature"].tolist(), title=None),
            color=alt.condition(alt.datum.coef > 0, alt.value(RED), alt.value(GREEN)),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("odds_ratio:Q", title="Odds ratio", format=".2f"),
            ],
        )
        .properties(height=24 * len(eff) + 20)
    )
    show_chart(chart)
    st.caption(
        "Odds ratio above 1 raises churn odds, below 1 lowers them. Features are "
        "standardised, so ratios are per one standard deviation."
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon=make_favicon(),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    missing = [p.name for p in ARTIFACT_PATHS if not p.exists()]
    if missing:
        st.error(
            f"Missing model files next to ui.py: {', '.join(missing)}. "
            "Run `python customer_churn.py` first."
        )
        st.stop()
    art = load_artifacts(_stamp(*ARTIFACT_PATHS))

    # Seed customer inputs once so every widget has a value on first run.
    for key, value in DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, value)

    top_bar(art)
    tab_prediction, tab_insights, tab_performance = st.tabs(TABS)
    with tab_prediction:
        page_prediction(art)
    with tab_insights:
        page_insights()
    with tab_performance:
        page_performance(art)

    st.markdown(
        '<p class="disclaimer">Scores are statistical estimates from historical '
        "data, intended to support (not replace) human judgement.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
