"""
Cardiac Risk Intelligence - Streamlit dashboard for the heart disease model.

Run from the heart_disease_prediction folder, after training:
    python main.py train --data heart_disease.csv
    streamlit run app.py

Reads everything main.py writes to ./artifacts (model.joblib, metrics.json,
metadata.json, data_quality.json, MODEL_CARD.md, evaluation.png) and reuses the
cleaning code in main.py, so the app and the training pipeline can never
disagree about how data is prepared.

Educational demonstration. Not a medical device.
"""

from __future__ import annotations

import os

# Keep numpy's BLAS thread pools small (avoids OpenBLAS memory errors on laptops).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import altair as alt  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score  # noqa: E402

import main as pipeline  # noqa: E402  (schema + cleaning code shared with training)

st.set_page_config(
    page_title="Cardiac Risk Intelligence",
    page_icon="\u2764\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("HEART_ARTIFACTS", APP_DIR / "artifacts"))

ACCENT, RISK, SAFE, MUTED = "#38bdf8", "#ef4444", "#22c55e", "#94a3b8"

PAGES = [
    "Risk assessment",
    "Batch scoring",
    "Data insights",
    "Model performance",
    "About",
]

# Every model family candidate_models() in main.py can produce.
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

FRIENDLY = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "thalch": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels (0-3)",
    "thal": "Thalassemia",
    "dataset": "Source site",
    "num": "Diagnosis (num)",
    "id": "Record id",
}

# Display label -> value used by the model (None = unknown / missing)
CP_OPTIONS = {
    "Typical angina": "typical angina",
    "Atypical angina": "atypical angina",
    "Non-anginal pain": "non-anginal",
    "Asymptomatic": "asymptomatic",
}
ECG_OPTIONS = {
    "Normal": "normal",
    "LV hypertrophy": "lv hypertrophy",
    "ST-T abnormality": "st-t abnormality",
}
SLOPE_OPTIONS = {
    "Upsloping": "upsloping",
    "Flat": "flat",
    "Downsloping": "downsloping",
    "Unknown": None,
}
THAL_OPTIONS = {
    "Normal": "normal",
    "Fixed defect": "fixed defect",
    "Reversible defect": "reversable defect",
    "Unknown": None,
}  # dataset spelling
VESSEL_OPTIONS = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "Unknown": None}
YESNO_OPTIONS = {"No": False, "Yes": True, "Unknown": None}
NUMERIC_LABELS = {
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "thalch": "Maximum heart rate",
    "oldpeak": "ST depression",
}

DEFAULT_PROFILE: dict[str, Any] = dict(
    age=54,
    sex="Male",
    cp="Asymptomatic",
    exang="No",
    fbs="No",
    trestbps=130,
    chol=240,
    restecg="Normal",
    thalch=150,
    oldpeak=1.0,
    slope="Flat",
    ca="Unknown",
    thal="Unknown",
    unavailable=[],
)
PRESETS: dict[str, dict[str, Any]] = {
    "Example: lower-risk profile": dict(
        age=51,
        sex="Female",
        cp="Atypical angina",
        exang="No",
        fbs="No",
        trestbps=130,
        chol=229,
        restecg="LV hypertrophy",
        thalch=168,
        oldpeak=0.0,
        slope="Upsloping",
        ca="0",
        thal="Normal",
        unavailable=[],
    ),
    "Example: higher-risk profile": dict(
        age=68,
        sex="Male",
        cp="Asymptomatic",
        exang="Yes",
        fbs="No",
        trestbps=150,
        chol=260,
        restecg="Normal",
        thalch=115,
        oldpeak=2.4,
        slope="Flat",
        ca="2",
        thal="Reversible defect",
        unavailable=[],
    ),
}
PROFILE_KEYS = list(DEFAULT_PROFILE)

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px;}
.hero {background: linear-gradient(115deg, #0b1f3a 0%, #0e3a5f 55%, #0b6b8f 100%);
  border: 1px solid rgba(56,189,248,.28); border-radius: 16px; padding: 1.5rem 1.8rem; margin-bottom: 1.4rem;}
.hero-eyebrow {color: #7dd3fc; font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 600;}
.hero-title {color: #f8fafc; font-size: 1.85rem; font-weight: 700; line-height: 1.25; margin: .25rem 0 .3rem;}
.hero-sub {color: #cbd5e1; font-size: .98rem; margin: 0;}
.section-label {font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: #94a3b8; margin: .3rem 0 .6rem;}
.brand {font-size: 1.15rem; font-weight: 700; line-height: 1.2;}
.brand-sub {font-size: .78rem; opacity: .65; margin-bottom: .4rem;}
.risk-card {border: 1px solid rgba(148,163,184,.28); border-left-width: 6px; border-radius: 14px;
  padding: 1.2rem 1.5rem 1rem; background: rgba(148,163,184,.07); margin-bottom: 1rem;}
.risk-head {display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;}
.pill {display: inline-block; padding: .22rem .8rem; border-radius: 999px; font-weight: 700; font-size: .82rem;}
.risk-sub {font-size: .84rem; opacity: .7;}
.risk-value {font-size: 3.4rem; font-weight: 800; line-height: 1.05; margin: .55rem 0 .9rem;}
.track {position: relative; height: 12px; border-radius: 6px; background: rgba(148,163,184,.25);}
.fill {height: 100%; border-radius: 6px;}
.marker {position: absolute; top: -6px; width: 3px; height: 24px; border-radius: 2px;
  background: currentColor; box-shadow: 0 0 0 1px rgba(0,0,0,.35);}
.scale {position: relative; height: 1.5rem; margin-top: .55rem; font-size: .76rem; opacity: .75;}
.scale span {position: absolute; white-space: nowrap;}
[data-testid="stMetric"] {background: rgba(148,163,184,.07); border: 1px solid rgba(148,163,184,.25);
  border-radius: 12px; padding: .8rem 1rem;}
.note {border: 1px solid rgba(245,158,11,.4); background: rgba(245,158,11,.08); border-radius: 10px;
  padding: .65rem .9rem; font-size: .84rem; margin-top: .6rem;}
</style>
"""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def html(markup: str) -> None:
    """Render HTML; strips indentation so markdown never turns it into a code block."""
    st.markdown(
        "\n".join(line.strip() for line in markup.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def site_label(name: str) -> str:
    return {"va long beach": "VA Long Beach"}.get(name, name.title())


def model_label(key: str | None) -> str:
    if not key:
        return "n/a"
    return MODEL_NAMES.get(key, str(key).replace("_", " ").title())


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    html(f"""<div class="hero"><div class="hero-eyebrow">{eyebrow}</div>
    <div class="hero-title">{title}</div><p class="hero-sub">{subtitle}</p></div>""")


def section(label: str) -> None:
    html(f'<div class="section-label">{label}</div>')


def disclaimer() -> None:
    html("""<div class="note"><b>Educational demonstration.</b> Trained on the public UCI Heart Disease
    dataset. This is not a medical device and does not replace professional clinical judgement.</div>""")


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_bundle(path: str, mtime: float) -> dict[str, Any]:
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_artifacts() -> dict[str, Any] | None:
    model_path = ARTIFACT_DIR / "model.joblib"
    if not model_path.exists():
        return None
    out: dict[str, Any] = {"bundle": _load_bundle(str(model_path), _mtime(model_path))}
    for name in ("metrics", "metadata", "data_quality"):
        path = ARTIFACT_DIR / f"{name}.json"
        out[name] = _load_json(str(path), _mtime(path)) if path.exists() else {}
    card = ARTIFACT_DIR / "MODEL_CARD.md"
    out["model_card"] = _load_text(str(card), _mtime(card)) if card.exists() else ""
    return out


def version_warning(bundle: dict[str, Any]) -> str | None:
    """Same check run_predict() performs in main.py."""
    trained = bundle.get("sklearn_version")
    if trained and trained != sklearn.__version__:
        return (
            f"This model was trained with scikit-learn {trained} but the app is running "
            f"{sklearn.__version__}. Results may differ; retrain to be safe."
        )
    return None


def score(bundle: dict[str, Any], raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    X = pipeline.prepare_features(
        raw, include_site=bundle["include_site"], strict=False
    )
    X = X[bundle["features"]]
    return bundle["pipeline"].predict_proba(X)[:, 1], X


def feature_influence(bundle: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Occlusion explanation: how much does the estimate drop if one input were unknown?"""
    base = float(bundle["pipeline"].predict_proba(X)[0, 1])
    variants = []
    for col in X.columns:
        variant = X.copy()
        variant.iloc[0, variant.columns.get_loc(col)] = np.nan
        variants.append(variant)
    masked = bundle["pipeline"].predict_proba(pd.concat(variants, ignore_index=True))[
        :, 1
    ]
    return pd.DataFrame(
        {
            "feature": [FRIENDLY.get(c, c) for c in X.columns],
            "shift_pp": 100 * (base - masked),
        }
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def influence_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.assign(
        direction=np.where(df["shift_pp"] >= 0, "Raises estimate", "Lowers estimate")
    )
    d = d.reindex(d["shift_pp"].abs().sort_values(ascending=False).index)
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "shift_pp:Q",
                title="Change in estimated probability (percentage points)",
            ),
            y=alt.Y("feature:N", sort=d["feature"].tolist(), title=None),
            color=alt.Color(
                "direction:N",
                title=None,
                legend=alt.Legend(orient="bottom"),
                scale=alt.Scale(
                    domain=["Raises estimate", "Lowers estimate"], range=[RISK, ACCENT]
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Input"),
                alt.Tooltip("shift_pp:Q", title="Shift (pp)", format="+.1f"),
            ],
        )
        .properties(height=max(220, 26 * len(d)))
    )


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_title: str,
    color: str = ACCENT,
    fmt: str = ".3f",
    height: int | None = None,
) -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort=df[y].tolist(), title=None),
        x=alt.X(f"{x}:Q", title=x_title),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)],
    )
    chart = base.mark_bar(color=color, cornerRadiusEnd=3) + base.mark_text(
        align="left", dx=4, color=MUTED
    ).encode(text=alt.Text(f"{x}:Q", format=fmt))
    return chart.properties(height=height or max(160, 30 * len(df)))


# --------------------------------------------------------------------------- #
# Page: risk assessment
# --------------------------------------------------------------------------- #
def init_state() -> None:
    for key, value in DEFAULT_PROFILE.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset", "Manual entry")
    # Re-assigning keeps widget values alive when the user switches pages.
    for key in PROFILE_KEYS + ["preset", "site"]:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def apply_preset() -> None:
    choice = st.session_state.get("preset")
    if choice in PRESETS:
        for key, value in PRESETS[choice].items():
            st.session_state[key] = value


def reset_form() -> None:
    for key, value in DEFAULT_PROFILE.items():
        st.session_state[key] = value
    st.session_state["preset"] = "Manual entry"


def patient_form(bundle: dict[str, Any]) -> None:
    st.selectbox(
        "Quick-fill an example",
        ["Manual entry"] + list(PRESETS),
        key="preset",
        on_change=apply_preset,
    )
    with st.container(border=True):
        section("Demographics")
        c1, c2 = st.columns(2)
        c1.slider("Age (years)", 18, 100, key="age")
        c2.selectbox("Sex", ["Male", "Female"], key="sex")
    with st.container(border=True):
        section("Symptoms and history")
        st.selectbox("Chest pain type", list(CP_OPTIONS), key="cp")
        c1, c2 = st.columns(2)
        c1.selectbox("Exercise-induced angina", list(YESNO_OPTIONS), key="exang")
        c2.selectbox("Fasting blood sugar > 120 mg/dL", list(YESNO_OPTIONS), key="fbs")
    with st.container(border=True):
        section("Vitals and laboratory")
        c1, c2 = st.columns(2)
        c1.slider("Resting blood pressure (mm Hg)", 80, 220, key="trestbps")
        c2.slider("Serum cholesterol (mg/dL)", 100, 600, key="chol")
        st.slider("Maximum heart rate achieved (bpm)", 60, 220, key="thalch")
    with st.container(border=True):
        section("ECG and imaging")
        c1, c2 = st.columns(2)
        c1.selectbox("Resting ECG", list(ECG_OPTIONS), key="restecg")
        c2.selectbox("ST slope at peak exercise", list(SLOPE_OPTIONS), key="slope")
        st.slider(
            "ST depression induced by exercise (mm)", -2.0, 6.5, step=0.1, key="oldpeak"
        )
        c1, c2 = st.columns(2)
        c1.selectbox(
            "Major vessels coloured by fluoroscopy", list(VESSEL_OPTIONS), key="ca"
        )
        c2.selectbox("Thalassemia (stress test)", list(THAL_OPTIONS), key="thal")
    if bundle["include_site"]:
        levels = bundle["categorical_levels"].get("dataset", [])
        st.selectbox(
            "Source site (this model was trained with the site as an input)",
            levels,
            key="site",
        )
    with st.expander("Some measurements are not available"):
        st.multiselect(
            "Mark as unavailable",
            list(NUMERIC_LABELS.values()),
            key="unavailable",
            help="The model handles missing values explicitly; it does not assume a number.",
        )
    st.button("Reset form", on_click=reset_form)


def profile_to_row(bundle: dict[str, Any]) -> dict[str, Any]:
    s = st.session_state
    off = set(s["unavailable"])

    def measured(key: str) -> float:
        return np.nan if NUMERIC_LABELS[key] in off else float(s[key])

    row = {
        "age": float(s["age"]),
        "sex": s["sex"].lower(),
        "cp": CP_OPTIONS[s["cp"]],
        "trestbps": measured("trestbps"),
        "chol": measured("chol"),
        "fbs": YESNO_OPTIONS[s["fbs"]],
        "restecg": ECG_OPTIONS[s["restecg"]],
        "thalch": measured("thalch"),
        "exang": YESNO_OPTIONS[s["exang"]],
        "oldpeak": measured("oldpeak"),
        "slope": SLOPE_OPTIONS[s["slope"]],
        "ca": VESSEL_OPTIONS[s["ca"]],
        "thal": THAL_OPTIONS[s["thal"]],
    }
    if bundle["include_site"]:
        row["dataset"] = s.get("site")
    return row


def risk_card(p: float, threshold: float) -> None:
    flagged = p >= threshold
    color = RISK if flagged else SAFE
    label = "Elevated risk" if flagged else "Lower risk"
    html(f"""<div class="risk-card" style="border-left-color:{color}">
    <div class="risk-head"><span class="pill" style="background:{color}26;color:{color}">{label}</span>
    <span class="risk-sub">Estimated probability of heart disease</span></div>
    <div class="risk-value" style="color:{color}">{100 * p:.1f}%</div>
    <div class="track"><div class="fill" style="width:{100 * p:.1f}%;background:{color}"></div>
    <div class="marker" style="left:calc({100 * threshold:.1f}% - 1px)"></div></div>
    <div class="scale"><span style="left:0">0%</span>
    <span style="left:{100 * threshold:.1f}%;transform:translateX(-50%)">screening threshold {100 * threshold:.0f}%</span>
    <span style="right:0">100%</span></div></div>""")


def driver_summary(infl: pd.DataFrame) -> str:
    up = infl.sort_values("shift_pp", ascending=False).iloc[0]
    down = infl.sort_values("shift_pp").iloc[0]
    parts = []
    if up["shift_pp"] >= 0.5:
        parts.append(
            f"largest upward influence **{up['feature']}** ({up['shift_pp']:+.1f} pp)"
        )
    if down["shift_pp"] <= -0.5:
        parts.append(
            f"largest downward influence **{down['feature']}** ({down['shift_pp']:+.1f} pp)"
        )
    if not parts:
        return "No single input moves the estimate by more than 0.5 points."
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def page_risk(state: dict[str, Any]) -> None:
    bundle, test = state["bundle"], state["metrics"].get("test", {})
    hero(
        "Risk assessment",
        "Individual risk assessment",
        "Enter a patient profile to estimate the probability of heart disease, "
        "together with a breakdown of what drives the estimate.",
    )
    left, right = st.columns([5, 6], gap="large")
    with left:
        patient_form(bundle)
    raw = pd.DataFrame([profile_to_row(bundle)])
    proba, X = score(bundle, raw)
    p, threshold = float(proba[0]), float(bundle["threshold"])
    infl = feature_influence(bundle, X)
    provided = int(X.notna().sum(axis=1).iloc[0])

    with right:
        risk_card(p, threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Inputs provided",
            f"{provided} / {X.shape[1]}",
            help="Missing inputs are handled by the model but make the estimate less reliable.",
        )
        c2.metric(
            "Model sensitivity",
            pct(test.get("recall")),
            help="Share of true disease cases flagged on held-out test patients.",
        )
        c3.metric(
            "Model specificity",
            pct(test.get("specificity")),
            help="Share of healthy patients correctly cleared on held-out test patients.",
        )
        min_recall = state["metadata"].get("config", {}).get("min_recall", 0.9)
        st.caption(
            f"The threshold ({pct(threshold, 0)}) was tuned on out-of-fold training predictions so that "
            f"at least {pct(min_recall, 0)} of disease cases are caught. "
            "That favours sensitivity, so false alarms are expected."
        )
        section("What drives this estimate")
        st.altair_chart(influence_chart(infl))
        st.markdown(driver_summary(infl))
        st.caption(
            "Each bar compares the estimate with the same patient if that single input were unknown. "
            "This is an approximate, model-based explanation, not a causal statement."
        )
        decision = "Elevated risk" if p >= threshold else "Lower risk"
        export = raw.assign(
            probability=round(p, 4),
            prediction=int(p >= threshold),
            decision=decision,
            threshold=round(threshold, 4),
        )
        st.download_button(
            "Download this assessment (CSV)",
            export.to_csv(index=False).encode("utf-8"),
            file_name="heart_risk_assessment.csv",
            mime="text/csv",
        )
        disclaimer()


# --------------------------------------------------------------------------- #
# Page: batch scoring
# --------------------------------------------------------------------------- #
def template_csv() -> bytes:
    rows = [
        {
            "id": 1,
            "age": 63,
            "sex": "Male",
            "cp": "typical angina",
            "trestbps": 145,
            "chol": 233,
            "fbs": True,
            "restecg": "lv hypertrophy",
            "thalch": 150,
            "exang": False,
            "oldpeak": 2.3,
            "slope": "downsloping",
            "ca": 0,
            "thal": "fixed defect",
        },
        {
            "id": 2,
            "age": 41,
            "sex": "Female",
            "cp": "atypical angina",
            "trestbps": 130,
            "chol": 204,
            "fbs": False,
            "restecg": "normal",
            "thalch": 172,
            "exang": False,
            "oldpeak": 1.4,
            "slope": "upsloping",
            "ca": 0,
            "thal": "normal",
        },
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def read_upload(upload: Any) -> pd.DataFrame:
    """CSV, or JSON (an object or a list of objects), like `main.py predict`."""
    if upload.name.lower().endswith(".json"):
        payload = json.load(upload)
        return pd.DataFrame([payload] if isinstance(payload, dict) else payload)
    return pd.read_csv(upload, na_values=["?"])


def upload_quality_report(
    raw: pd.DataFrame, X: pd.DataFrame, bundle: dict[str, Any]
) -> None:
    missing_cols = [c for c in pipeline.ALL_FEATURES if c not in raw.columns]
    if missing_cols:
        st.warning("Columns not found (treated as missing): " + ", ".join(missing_cols))

    issues = []
    for col in pipeline.NUMERIC_FEATURES:
        if col in raw.columns:
            values = pd.to_numeric(raw[col], errors="coerce")
            low, high = pipeline.PLAUSIBLE_RANGES[col]
            bad = int((values.notna() & ~values.between(low, high)).sum())
            if bad:
                issues.append(
                    {
                        "Column": FRIENDLY[col],
                        "Issue": f"{bad} value(s) outside [{low}, {high}] set to missing",
                    }
                )

    # Same check run_predict() does: categories the model never saw are encoded as all-zero.
    unseen_rows = []
    for col, known in bundle.get("categorical_levels", {}).items():
        if col in X.columns:
            unseen = set(X[col].dropna().unique()) - set(known)
            if unseen:
                unseen_rows.append(
                    {
                        "Column": FRIENDLY.get(col, col),
                        "Unseen values": ", ".join(sorted(map(str, unseen))),
                        "Known values": ", ".join(map(str, known)),
                    }
                )
    if unseen_rows:
        st.warning(
            "Some categories were not seen during training and are encoded as all-zero. "
            "Open the data quality report for details."
        )

    with st.expander("Data quality report"):
        rates = pd.DataFrame(
            {
                "Input": [FRIENDLY.get(c, c) for c in X.columns],
                "Missing": 100 * X.isna().mean().to_numpy(),
            }
        )
        st.dataframe(
            rates.sort_values("Missing", ascending=False),
            hide_index=True,
            column_config={
                "Missing": st.column_config.ProgressColumn(
                    "Missing", min_value=0.0, max_value=100.0, format="%.0f%%"
                )
            },
        )
        if issues:
            st.dataframe(pd.DataFrame(issues), hide_index=True)
        else:
            st.caption("No implausible values detected.")
        if unseen_rows:
            section("Unseen categories")
            st.dataframe(pd.DataFrame(unseen_rows), hide_index=True)


def page_batch(state: dict[str, Any]) -> None:
    bundle = state["bundle"]
    hero(
        "Batch scoring",
        "Score many patients at once",
        "Upload a CSV or JSON file of patient records to estimate risk for every row and export the results.",
    )
    c1, c2 = st.columns([3, 2], gap="large")
    upload = c1.file_uploader("Patient records (CSV or JSON)", type=["csv", "json"])
    c2.markdown("**Expected columns**")
    c2.caption(
        ", ".join(pipeline.ALL_FEATURES)
        + ". An optional `id` column is kept in the output. "
        "JSON can be a single object or a list of objects."
    )
    c2.download_button(
        "Download CSV template",
        template_csv(),
        file_name="patients_template.csv",
        mime="text/csv",
    )
    if upload is None:
        st.info(
            "Upload a file to begin. Missing values are fine; the model handles them explicitly."
        )
        return
    try:
        raw = read_upload(upload)
    except Exception as exc:  # noqa: BLE001 - surface any parsing problem to the user
        st.error(f"Could not read the file: {exc}")
        return
    if raw.empty or len(raw) > 20000:
        st.error("The file must contain between 1 and 20,000 rows.")
        return

    proba, X = score(bundle, raw)
    threshold = float(bundle["threshold"])
    flagged = proba >= threshold

    # Output mirrors `main.py predict` (id, probability, prediction) plus a readable decision.
    result = raw.drop(
        columns=[c for c in ("probability", "prediction", "decision") if c in raw.columns]
    ).copy()
    lead = 1 if "id" in result.columns else 0
    result.insert(lead, "probability", np.round(proba, 4))
    result.insert(lead + 1, "prediction", flagged.astype(int))
    result.insert(
        lead + 2, "decision", np.where(flagged, "Elevated risk", "Lower risk")
    )
    result = result.sort_values("probability", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records scored", f"{len(result):,}")
    k2.metric("Flagged elevated risk", f"{int(flagged.sum()):,}")
    k3.metric("Share flagged", pct(float(flagged.mean())))
    k4.metric("Median probability", pct(float(np.median(proba))))

    section("Distribution of estimated risk")
    hist_df = pd.DataFrame({"probability": proba})
    hist = (
        alt.Chart(hist_df)
        .mark_bar(color=ACCENT, opacity=0.85)
        .encode(
            x=alt.X(
                "probability:Q",
                bin=alt.Bin(maxbins=25),
                title="Estimated probability",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("count():Q", title="Patients"),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color=RISK, strokeDash=[6, 4], size=2)
        .encode(x="t:Q")
    )
    st.altair_chart((hist + rule).properties(height=240))
    st.caption(f"Dashed line: screening threshold ({pct(threshold, 0)}).")

    section("Results")
    st.dataframe(
        result,
        hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Probability", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.download_button(
        "Download results (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="heart_risk_predictions.csv",
        mime="text/csv",
    )
    upload_quality_report(raw, X, bundle)

    if pipeline.TARGET_COLUMN in raw.columns:
        try:
            y = pipeline.build_target(raw)
        except pipeline.DataValidationError:
            y = None
        if y is not None and y.nunique() == 2:
            section("Labelled data detected: how the model performs on this file")
            tn, fp, fn, tp = confusion_matrix(
                y, flagged.astype(int), labels=[0, 1]
            ).ravel()
            m1, m2, m3 = st.columns(3)
            m1.metric("ROC-AUC", num(float(roc_auc_score(y, proba))))
            m2.metric("Sensitivity", pct(float(recall_score(y, flagged.astype(int)))))
            m3.metric("Specificity", pct(float(tn / (tn + fp))) if (tn + fp) else "n/a")
            st.caption(
                "If these patients were part of the training data, these figures are optimistic. "
                "Use the Model performance page for the held-out results."
            )


# --------------------------------------------------------------------------- #
# Page: data insights
# --------------------------------------------------------------------------- #
def find_dataset(metadata: dict[str, Any]) -> Path | None:
    candidates = [APP_DIR / "heart_disease.csv", APP_DIR / "data" / "heart_disease.csv"]
    recorded = metadata.get("data", {}).get("path")
    if recorded:
        candidates += [Path(recorded), APP_DIR / recorded]
    return next((c for c in candidates if c.is_file()), None)


def trained_without_duplicates(metadata: dict[str, Any]) -> bool:
    """True when the model was trained with --drop-duplicate-rows."""
    return bool(metadata.get("config", {}).get("drop_duplicate_rows"))


@st.cache_data(show_spinner=False)
def load_insight_data(path: str, mtime: float, drop_duplicates: bool) -> pd.DataFrame:
    raw = pd.read_csv(path, na_values=["?"])
    if drop_duplicates:
        # Same rule as run_train(), so the explorer shows the rows the model learned from.
        subset = [
            c
            for c in pipeline.ALL_FEATURES + [pipeline.TARGET_COLUMN, pipeline.SITE_COLUMN]
            if c in raw.columns
        ]
        raw = raw.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    X = pipeline.prepare_features(raw, include_site=False, strict=True)
    X["target"] = pipeline.build_target(raw).to_numpy()
    X["Outcome"] = np.where(X["target"] == 1, "Disease", "No disease")
    if pipeline.SITE_COLUMN in raw.columns:
        X["site"] = [
            site_label(s)
            for s in pipeline._normalise_text(raw[pipeline.SITE_COLUMN])
            .fillna("unknown")
            .astype(str)
        ]
    return X


def render_data_quality(dq: dict[str, Any]) -> None:
    """The audit main.py writes to data_quality.json before any modelling."""
    if not dq:
        st.info(
            "data_quality.json was not found next to the model. Retrain to generate the audit."
        )
        return
    n_rows = int(dq.get("n_rows", 0))
    implausible = {
        k: v for k, v in dq.get("implausible_numeric_values", {}).items() if v
    }
    k = st.columns(4)
    k[0].metric("Rows x columns", f"{n_rows:,} x {dq.get('n_columns', 0)}")
    k[1].metric(
        "Exact duplicate rows",
        f"{dq.get('exact_duplicate_rows', 0):,}",
        help="Identical in every column. Train with --drop-duplicate-rows to remove them.",
    )
    k[2].metric(
        "Duplicate feature rows",
        f"{dq.get('duplicate_feature_rows', 0):,}",
        help="Identical across the cleaned model inputs, whatever the outcome.",
    )
    k[3].metric(
        "Implausible values",
        f"{sum(implausible.values()):,}",
        help="Numeric values outside the plausible range; the model treats them as missing.",
    )

    a, b = st.columns(2, gap="large")
    with a:
        section("Implausible values set to missing")
        if implausible:
            ranges = pipeline.PLAUSIBLE_RANGES
            st.dataframe(
                pd.DataFrame(
                    {
                        "Input": [FRIENDLY.get(c, c) for c in implausible],
                        "Plausible range": [
                            f"{ranges[c][0]} to {ranges[c][1]}" if c in ranges else "n/a"
                            for c in implausible
                        ],
                        "Values": list(implausible.values()),
                    }
                ),
                hide_index=True,
            )
        else:
            st.caption("No implausible values found.")
        section("Mixed-type columns")
        mixed = dq.get("mixed_type_columns", [])
        if mixed:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": [m["column"] for m in mixed],
                        "Types found": [
                            ", ".join(
                                t.replace("<class '", "").replace("'>", "")
                                for t in m.get("python_types", [])
                            )
                            for m in mixed
                        ],
                    }
                ),
                hide_index=True,
            )
        else:
            st.caption("Every column holds a single data type.")
    with b:
        section("Missing values by column (raw file)")
        missing = {c: v for c, v in dq.get("missing_values", {}).items() if v}
        if missing and n_rows:
            miss = pd.DataFrame(
                {
                    "Input": [FRIENDLY.get(c, c) for c in missing],
                    "Missing": [v / n_rows for v in missing.values()],
                }
            ).sort_values("Missing", ascending=False)
            st.altair_chart(bar_chart(miss, "Missing", "Input", "Share missing", fmt=".0%"))
        else:
            st.caption("No missing values in the raw file.")
        counts = dq.get("target_counts", {})
        if counts:
            st.caption(
                f"Outcome counts: {counts.get('0', 0):,} without disease, "
                f"{counts.get('1', 0):,} with disease "
                f"({pct(dq.get('target_positive_rate'))} positive)."
            )


def page_insights(state: dict[str, Any]) -> None:
    meta, dq, bundle = state["metadata"], state["data_quality"], state["bundle"]
    hero(
        "Data insights",
        "The cohort behind the model",
        "Outcomes, source sites and data quality in the UCI Heart Disease dataset used for training.",
    )
    path = find_dataset(meta)
    if path is None:
        info = meta.get("data", {})
        if not info and not dq:
            st.info(
                "Place heart_disease.csv next to app.py to unlock the interactive explorer."
            )
            return
        st.info(
            "heart_disease.csv was not found next to app.py, so this page shows the summary saved at training time."
        )
        if info:
            k1, k2, k3 = st.columns(3)
            k1.metric("Patients", f"{info.get('n_rows', 0):,}")
            k2.metric("Disease prevalence", pct(info.get("positive_rate")))
            k3.metric("Duplicate feature rows", f"{info.get('duplicate_feature_rows', 0):,}")
            miss = pd.DataFrame(
                {
                    "Input": [FRIENDLY.get(k, k) for k in info.get("missing_rate", {})],
                    "Missing": list(info.get("missing_rate", {}).values()),
                }
            )
            if not miss.empty:
                section("Missing values by input")
                st.altair_chart(
                    bar_chart(
                        miss.sort_values("Missing", ascending=False),
                        "Missing",
                        "Input",
                        "Share missing",
                        fmt=".0%",
                    )
                )
            rates = info.get("positive_rate_per_site", {})
            counts = info.get("rows_per_site", {})
            if rates:
                sites = pd.DataFrame(
                    {
                        "label": [
                            f"{site_label(s)}  (n={counts.get(s, '?')})" for s in rates
                        ],
                        "rate": list(rates.values()),
                    }
                ).sort_values("rate")
                section("Disease rate by source site")
                st.altair_chart(bar_chart(sites, "rate", "label", "Disease rate", fmt=".0%"))
        section("Data quality audit")
        render_data_quality(dq)
        return

    df = load_insight_data(str(path), _mtime(path), trained_without_duplicates(meta))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients", f"{len(df):,}")
    k2.metric("Disease prevalence", pct(float(df["target"].mean())))
    k3.metric("Source sites", df["site"].nunique() if "site" in df else "n/a")
    k4.metric("Model inputs", len(pipeline.ALL_FEATURES))
    if trained_without_duplicates(meta):
        st.caption("Exact duplicate rows are removed here, as they were during training.")

    colors = alt.Scale(domain=["No disease", "Disease"], range=[ACCENT, RISK])
    a, b = st.columns(2, gap="large")
    with a:
        section("Distribution by outcome")
        feature = st.selectbox(
            "Measurement", pipeline.NUMERIC_FEATURES, format_func=lambda c: FRIENDLY[c]
        )
        chart = (
            alt.Chart(df[[feature, "Outcome"]].dropna())
            .mark_bar(opacity=0.65)
            .encode(
                x=alt.X(
                    f"{feature}:Q", bin=alt.Bin(maxbins=25), title=FRIENDLY[feature]
                ),
                y=alt.Y("count():Q", stack=None, title="Patients"),
                color=alt.Color(
                    "Outcome:N",
                    scale=colors,
                    legend=alt.Legend(orient="bottom", title=None),
                ),
            )
        )
        st.altair_chart(chart.properties(height=280))
    with b:
        section("Disease rate by category")
        cats = [c for c in pipeline.CATEGORICAL_FEATURES + pipeline.BINARY_FEATURES]
        category = st.selectbox(
            "Characteristic", cats, format_func=lambda c: FRIENDLY[c]
        )
        labels = (
            df[category].map(lambda v: {1.0: "Yes", 0.0: "No"}.get(v, v))
            if category in pipeline.BINARY_FEATURES
            else df[category]
        )
        grouped = (
            df.assign(group=labels.fillna("missing").astype(str))
            .groupby("group")["target"]
            .agg(rate="mean", n="count")
            .reset_index()
            .sort_values("rate", ascending=False)
        )
        base = alt.Chart(grouped).encode(
            y=alt.Y("group:N", sort=grouped["group"].tolist(), title=None),
            x=alt.X(
                "rate:Q",
                title="Disease rate",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("rate:Q", format=".0%"),
                alt.Tooltip("n:Q", title="Patients"),
            ],
        )
        st.altair_chart(
            (
                base.mark_bar(color=ACCENT, cornerRadiusEnd=3)
                + base.mark_text(align="left", dx=4, color=MUTED).encode(
                    text=alt.Text("rate:Q", format=".0%")
                )
            ).properties(height=max(160, 40 * len(grouped)))
        )
        st.caption("Groups with few patients can swing widely; hover for counts.")

    c, d = st.columns(2, gap="large")
    with c:
        section("Disease rate by source site")
        if "site" in df:
            sites = (
                df.groupby("site")["target"]
                .agg(rate="mean", n="count")
                .reset_index()
                .sort_values("rate")
            )
            sites["label"] = sites["site"] + "  (n=" + sites["n"].astype(str) + ")"
            st.altair_chart(
                bar_chart(sites, "rate", "label", "Disease rate", fmt=".0%")
            )
            if bundle["include_site"]:
                st.caption(
                    "Rates differ sharply between hospitals. This model was trained with the site as an "
                    "input (--include-site), so these differences feed directly into its estimates."
                )
            else:
                st.caption(
                    "Rates differ sharply between hospitals, which is why the model does not use the site as an input."
                )
    with d:
        section("Missing values after cleaning")
        feats = [c for c in pipeline.ALL_FEATURES]
        miss = pd.DataFrame(
            {
                "Input": [FRIENDLY[c] for c in feats],
                "Missing": df[feats].isna().mean().to_numpy(),
            }
        )
        st.altair_chart(
            bar_chart(
                miss.sort_values("Missing", ascending=False),
                "Missing",
                "Input",
                "Share missing",
                fmt=".0%",
            )
        )
        st.caption(
            "Implausible values (for example cholesterol = 0) are counted as missing."
        )

    section("Data quality audit (recorded at training time)")
    render_data_quality(dq)


# --------------------------------------------------------------------------- #
# Page: model performance
# --------------------------------------------------------------------------- #
def table_from_groups(groups: dict[str, Any], keep: list[str]) -> pd.DataFrame:
    rows = []
    for name, vals in groups.items():
        row = {"Group": site_label(name)}
        row.update({k: vals.get(k) for k in keep})
        row["Note"] = vals.get("note")  # e.g. "skipped (n < 20)" from subgroup_metrics
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df["Note"].isna().all():
        df = df.drop(columns="Note")
    return df


def importance_chart(imp: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(imp).encode(
        y=alt.Y("Input:N", sort=imp["Input"].tolist(), title=None)
    )
    bars = base.mark_bar(color=ACCENT, cornerRadiusEnd=3).encode(
        x=alt.X(
            "mean_drop_in_auc:Q", title="Drop in ROC-AUC when the input is shuffled"
        ),
        tooltip=[
            alt.Tooltip("Input:N"),
            alt.Tooltip("mean_drop_in_auc:Q", title="Mean drop", format=".4f"),
            alt.Tooltip("std:Q", title="Std", format=".4f"),
        ],
    )
    if "std" not in imp:
        return bars.properties(height=max(220, 28 * len(imp)))
    whiskers = base.mark_rule(color=MUTED).encode(x="low:Q", x2="high:Q")
    return (bars + whiskers).properties(height=max(220, 28 * len(imp)))


def page_performance(state: dict[str, Any]) -> None:
    metrics, meta = state["metrics"], state["metadata"]
    hero(
        "Model performance",
        "Held-out evaluation",
        "Results on patients the model never saw during training or tuning, plus robustness checks.",
    )
    test = metrics.get("test")
    if not test:
        st.warning(
            "metrics.json was not found. Run `python main.py train --data heart_disease.csv`."
        )
        return
    ci = test.get("roc_auc_95ci")
    pr_ci = test.get("pr_auc_95ci")
    prevalence = test.get("prevalence")
    brier_ref = prevalence * (1 - prevalence) if prevalence is not None else None
    k = st.columns(5)
    k[0].metric(
        "ROC-AUC",
        num(test.get("roc_auc")),
        help=f"95% bootstrap interval: {num(ci[0])} to {num(ci[1])}" if ci else None,
    )
    k[1].metric(
        "PR-AUC",
        num(test.get("pr_auc")),
        help=f"95% bootstrap interval: {num(pr_ci[0])} to {num(pr_ci[1])}"
        if pr_ci
        else None,
    )
    k[2].metric("Sensitivity", pct(test.get("recall")))
    k[3].metric("Specificity", pct(test.get("specificity")))
    k[4].metric(
        "Brier score",
        num(test.get("brier")),
        help="Lower is better; measures probability accuracy."
        + (
            f" Giving every patient the base rate scores {num(brier_ref)}."
            if brier_ref is not None
            else ""
        ),
    )
    st.caption(
        f"{test['n']} held-out patients | disease prevalence {pct(test['prevalence'])} | "
        f"selected model: {model_label(metrics.get('selected_model'))} | "
        f"threshold {num(metrics.get('threshold'))}"
    )

    t1, t2, t3, t4 = st.tabs(
        ["Evaluation", "Model comparison", "Robustness", "Feature importance"]
    )
    with t1:
        a, b = st.columns([3, 2], gap="large")
        image = ARTIFACT_DIR / "evaluation.png"
        if image.exists():
            a.image(str(image))
            a.caption(
                "ROC and precision-recall curves, calibration and the confusion matrix on the held-out test set."
            )
        with b:
            section("Confusion matrix")
            cm = pd.DataFrame(
                [[test["tn"], test["fp"]], [test["fn"], test["tp"]]],
                index=["Actual: no disease", "Actual: disease"],
                columns=["Predicted: no disease", "Predicted: disease"],
            )
            st.dataframe(cm)
            st.markdown(
                f"- Accuracy: **{pct(test.get('accuracy'))}**\n"
                f"- Balanced accuracy: **{pct(test.get('balanced_accuracy'))}**\n"
                f"- Precision: **{pct(test.get('precision'))}**\n"
                f"- F1 score: **{num(test.get('f1'))}**\n"
                f"- Matthews correlation (MCC): **{num(test.get('mcc'))}**\n"
                f"- Majority-class baseline accuracy: **{pct(metrics.get('baseline', {}).get('majority_class_accuracy'))}**"
            )
            st.caption(metrics.get("threshold_policy", ""))
    with t2:
        sel = metrics.get("model_selection", {})
        chosen = metrics.get("selected_model")
        rows = [
            {
                "Model": model_label(k),
                "Selected": "\u2713" if k == chosen else "",
                "CV ROC-AUC": v["cv_roc_auc_mean"],
                "Std": v["cv_roc_auc_std"],
                "Best parameters": ", ".join(
                    f"{p}={val:.3g}" if isinstance(val, float) else f"{p}={val}"
                    for p, val in v["best_params"].items()
                ),
            }
            for k, v in sel.items()
        ]
        if rows:
            comp = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False)
            st.altair_chart(
                bar_chart(
                    comp.rename(columns={"CV ROC-AUC": "auc"}),
                    "auc",
                    "Model",
                    "Cross-validated ROC-AUC (training data)",
                )
            )
            st.dataframe(comp, hide_index=True)
        else:
            st.info("No model comparison was saved with this model.")
        st.caption(
            "Each family was tuned with randomized search and stratified cross-validation on the training split only."
        )
        if "xgboost" not in sel and not meta.get("environment", {}).get(
            "xgboost_available", False
        ):
            st.caption(
                "XGBoost was not installed when this model was trained, so it was not compared. "
                "Run `pip install xgboost` and retrain to include it."
            )
    with t3:
        sub = metrics.get("subgroups", {})
        keep = ["n", "prevalence", "roc_auc", "recall", "specificity"]
        cfg = {
            "n": st.column_config.NumberColumn("Patients"),
            "prevalence": st.column_config.NumberColumn("Prevalence", format="%.2f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
            "recall": st.column_config.NumberColumn("Sensitivity", format="%.2f"),
            "specificity": st.column_config.NumberColumn("Specificity", format="%.2f"),
        }
        if sub.get("sex"):
            section("Test performance by sex")
            st.dataframe(
                table_from_groups(sub["sex"], keep), hide_index=True, column_config=cfg
            )
        if sub.get("site"):
            section("Test performance by source site")
            st.dataframe(
                table_from_groups(sub["site"], keep), hide_index=True, column_config=cfg
            )
            st.caption(
                "Groups with fewer than 20 test patients are listed but not scored."
            )
        loso = metrics.get("leave_one_site_out")
        if loso:
            section("Leave-one-site-out validation")
            st.dataframe(
                table_from_groups(
                    loso,
                    ["n", "prevalence", "roc_auc", "pr_auc", "recall", "specificity"],
                ),
                hide_index=True,
                column_config={
                    **cfg,
                    "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                },
            )
            st.caption(
                "Each hospital is held out in turn and predicted by a model trained on the other three. "
                "Hyper-parameters were tuned on pooled data, so treat this as a slightly optimistic transfer check."
            )
        if not (sub or loso):
            st.info("No robustness results were saved with this model.")
    with t4:
        imp = pd.DataFrame(metrics.get("feature_importance", []))
        if imp.empty:
            st.info("No feature importance was saved with this model.")
        else:
            imp["Input"] = imp["feature"].map(lambda c: FRIENDLY.get(c, c))
            imp = imp.sort_values("mean_drop_in_auc", ascending=False)
            if "std" in imp:
                imp["low"] = imp["mean_drop_in_auc"] - imp["std"]
                imp["high"] = imp["mean_drop_in_auc"] + imp["std"]
            st.altair_chart(importance_chart(imp))
            st.caption(
                "Permutation importance on the held-out test set: how much accuracy is lost when one input is "
                "scrambled. Whiskers show one standard deviation across repeated shuffles."
            )


# --------------------------------------------------------------------------- #
# Page: about
# --------------------------------------------------------------------------- #
def page_about(state: dict[str, Any]) -> None:
    meta, metrics, bundle = state["metadata"], state["metrics"], state["bundle"]
    cfg = meta.get("config", {})
    data, env = meta.get("data", {}), meta.get("environment", {})
    hero(
        "About",
        "How the model was built",
        "Data preparation, validation approach and the limits of what this tool can tell you.",
    )

    families = (
        ", ".join(model_label(k) for k in metrics.get("model_selection", {}))
        or "logistic regression, random forest and gradient boosting"
    )
    min_recall = cfg.get("min_recall", 0.9)
    duplicates = (
        "exact duplicate rows were removed before splitting"
        if cfg.get("drop_duplicate_rows")
        else "duplicates are reported but kept (train with `--drop-duplicate-rows` to remove them)"
    )
    evaluation = (
        "one pass on a held-out test set with bootstrap confidence intervals, calibration, "
        "permutation importance and subgroup checks"
        + (
            ", plus leave-one-site-out validation"
            if metrics.get("leave_one_site_out")
            else ""
        )
    )
    n_rows, n_sites = data.get("n_rows"), len(data.get("rows_per_site", {}))
    cohort = (
        f"Trained on {n_rows:,} patients"
        + (f" from {n_sites} hospitals" if n_sites else "")
        if isinstance(n_rows, int)
        else "Trained on roughly 900 patients from four hospitals"
    )

    a, b = st.columns(2, gap="large")
    with a:
        section("Method")
        st.markdown(
            "1. **Target:** angiographic disease present (`num > 0`) versus absent.\n"
            f"2. **Data audit:** missing values, mixed-type fields, implausible values and duplicates are checked before modelling; {duplicates}.\n"
            "3. **Cleaning:** types are coerced and physiologically impossible values (for example cholesterol of 0) become missing.\n"
            "4. **Leakage control:** imputation, scaling and encoding are fitted inside cross-validation folds only.\n"
            f"5. **Selection:** {families} are tuned by randomized search on ROC-AUC "
            f"({cfg.get('n_iter', 'n/a')} iterations each, {cfg.get('cv_folds', 'n/a')}-fold stratified cross-validation).\n"
            f"6. **Threshold:** chosen from out-of-fold predictions to keep sensitivity at or above {pct(min_recall, 0)} with the fewest false alarms.\n"
            f"7. **Evaluation:** {evaluation}."
        )
        section("Intended use and limits")
        st.markdown(
            "- Educational and portfolio demonstration of an end-to-end ML workflow.\n"
            f"- {cohort}, collected decades ago; it may not generalise to other populations.\n"
            "- The decision threshold favours sensitivity, so many flagged patients will not have disease.\n"
            + (
                "- The source site is a model input, so estimates depend on which hospital a profile resembles.\n"
                if bundle["include_site"]
                else "- The source site is not a model input; performance still varies between hospitals.\n"
            )
            + "- Not validated for clinical use. Never use it to make or delay medical decisions."
        )
    with b:
        section("Reproducibility")
        trained = meta.get("created_at_utc")
        if trained:
            try:
                trained = datetime.fromisoformat(trained).strftime(
                    "%d %b %Y, %H:%M UTC"
                )
            except ValueError:
                pass
        lines = [
            f"- **Trained:** {trained or 'n/a'}",
            f"- **Git commit:** `{meta.get('git_commit') or 'n/a'}`",
            f"- **Training rows:** {data.get('n_rows', 'n/a')}",
            f"- **Data fingerprint (SHA-256):** `{str(data.get('sha256', 'n/a'))[:16]}...`",
            f"- **Seed:** {cfg.get('seed', 'n/a')}",
            f"- **Split and search:** {pct(cfg.get('test_size'), 0)} test, "
            f"{cfg.get('cv_folds', 'n/a')}-fold CV, {cfg.get('n_iter', 'n/a')} iterations per model, "
            f"{cfg.get('n_bootstrap', 'n/a')} bootstrap resamples",
            f"- **Python / scikit-learn:** {env.get('python', 'n/a')} / {env.get('scikit_learn', 'n/a')}",
            f"- **pandas / numpy:** {env.get('pandas', 'n/a')} / {env.get('numpy', 'n/a')}",
            f"- **Platform:** {env.get('platform', 'n/a')}",
            f"- **Optional packages:** XGBoost {'installed' if env.get('xgboost_available') else 'not installed'}, "
            f"MLflow {'installed' if env.get('mlflow_available') else 'not installed'}",
        ]
        if metrics.get("mlflow_run_id"):
            lines.append(f"- **MLflow run:** `{metrics['mlflow_run_id']}`")
        st.markdown("\n".join(lines))
        warning = version_warning(bundle)
        if warning:
            st.warning(warning)
        section("Retrain and score from the command line")
        st.code(
            "python main.py train --data heart_disease.csv\n"
            "# optional flags: --drop-duplicate-rows  --include-site  --track-mlflow  --n-iter 10  --n-jobs 1\n"
            "python main.py predict --input patients.csv --output predictions.csv",
            language="powershell",
        )
        card = state.get("model_card", "")
        with st.expander("Model card"):
            if card:
                st.markdown(card)
                st.download_button(
                    "Download model card (Markdown)",
                    card.encode("utf-8"),
                    file_name="MODEL_CARD.md",
                    mime="text/markdown",
                )
            else:
                st.caption("MODEL_CARD.md was not found. Retrain to generate it.")
        with st.expander("Raw metadata"):
            st.json(meta, expanded=False)
    disclaimer()


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def sidebar(state: dict[str, Any]) -> str:
    with st.sidebar:
        html(
            '<div class="brand">Cardiac Risk Intelligence</div>'
            '<div class="brand-sub">Machine-learning decision support</div>'
        )
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        metrics, meta, bundle = state["metrics"], state["metadata"], state["bundle"]
        section("Active model")
        name = model_label(metrics.get("selected_model") or bundle.get("model_name"))
        trained = meta.get("created_at_utc", "")[:10] or "n/a"
        st.markdown(f"**{name}**")
        st.caption(
            f"ROC-AUC {num(metrics.get('test', {}).get('roc_auc'))} on held-out data  \n"
            f"Threshold {num(bundle['threshold'])}  \nTrained {trained}"
        )
        warning = version_warning(bundle)
        if warning:
            st.warning(warning)
        st.divider()
        st.caption("Educational demonstration. Not a medical device.")
    return page


def missing_artifacts() -> None:
    hero(
        "Setup required",
        "No trained model found",
        "The dashboard reads the files created by main.py. Train the model first.",
    )
    st.code(
        "python main.py train --data heart_disease.csv\nstreamlit run app.py",
        language="powershell",
    )
    st.caption(f"Looking for: {ARTIFACT_DIR / 'model.joblib'}")


def run() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    state = load_artifacts()
    if state is None:
        missing_artifacts()
        return
    init_state()
    page = sidebar(state)
    {
        "Risk assessment": page_risk,
        "Batch scoring": page_batch,
        "Data insights": page_insights,
        "Model performance": page_performance,
        "About": page_about,
    }[page](state)


run()