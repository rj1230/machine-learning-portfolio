"""
Emotion Intelligence - Streamlit Application
============================================

A professional workspace for analyzing the emotional tone of text.

Pages:
- Analyze text
- Batch analysis
- Insights
- About & limitations

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "bidirectional_gru_model.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"

# Must exactly match EXPECTED_CLASS_ORDER in main.py.
CLASS_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

# Must exactly match MAX_SEQUENCE_LENGTH in main.py.
MAX_SEQUENCE_LENGTH = 50

MAX_INPUT_CHARACTERS = 1000
MAX_BATCH_ROWS = 500
UNCERTAIN_COLOR = "#8F8878"
PRIMARY_COLOR = "#CDA95B"

CLASS_METADATA = {
    "sadness": {
        "emoji": "😢",
        "color": "#6F93C9",
        "description": "Low mood, disappointment, grief, loneliness, or hopelessness.",
    },
    "joy": {
        "emoji": "😄",
        "color": "#7DBE6E",
        "description": "Happiness, excitement, gratitude, satisfaction, or optimism.",
    },
    "love": {
        "emoji": "❤️",
        "color": "#E26D8A",
        "description": "Affection, appreciation, care, attachment, or romantic emotion.",
    },
    "anger": {
        "emoji": "😠",
        "color": "#E0583A",
        "description": "Frustration, irritation, resentment, annoyance, or hostility.",
    },
    "fear": {
        "emoji": "😨",
        "color": "#A585D6",
        "description": "Worry, anxiety, nervousness, uncertainty, or feeling threatened.",
    },
    "surprise": {
        "emoji": "😮",
        "color": "#3FB8AF",
        "description": "Unexpected events, shock, amazement, curiosity, or disbelief.",
    },
}

EXAMPLE_TEXTS = {
    "Joyful statement": "I feel so happy and grateful today.",
    "Sad statement": "I feel so alone and hopeless today.",
    "Angry statement": "I am furious that they cancelled the trip at the last minute.",
    "Fearful statement": "I feel terrified when walking down dark alleyways alone.",
    "Surprised statement": "I was shocked and completely surprised by the unexpected gift!",
    "Loving statement": "I really appreciate everything you do for me and care about you deeply.",
    "Neutral statement": "The meeting is scheduled for tomorrow morning at ten o'clock.",
}

SAMPLE_CSV = """text
I am extremely happy with the support I received today.
This is unacceptable. I have been waiting for three weeks.
I feel nervous about the upcoming presentation.
Thank you so much for your help. You made my day.
I was surprised by how quickly the issue was resolved.
I feel lonely and discouraged lately.
The invoice was issued on Monday.
"""


# ---------------------------------------------------------------------------
# Model result objects
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    text: str
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    token_count: int


# ---------------------------------------------------------------------------
# Emotion classifier
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading emotion model...")
def load_classifier() -> "EmotionClassifier":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n\n"
            "Run `python main.py` first to train and export the model."
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}\n\n"
            "Run `python main.py` first to train and export the tokenizer."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    return EmotionClassifier(model, tokenizer)


class EmotionClassifier:
    """Loads the exported Bidirectional GRU model and tokenizer."""

    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        sequences = self.tokenizer.texts_to_sequences(texts)

        return pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )

    def predict(self, texts: list[str]) -> list[PredictionResult]:
        encoded = self.encode(texts)
        probabilities = self.model.predict(encoded, verbose=0)

        results = []

        for text, probability_row in zip(texts, probabilities):
            probabilities = {
                name: float(probability_row[index])
                for index, name in enumerate(CLASS_NAMES)
            }

            top_index = int(np.argmax(probability_row))
            token_count = len(self.tokenizer.texts_to_sequences([text])[0])

            results.append(
                PredictionResult(
                    text=text,
                    emotion=CLASS_NAMES[top_index],
                    confidence=float(probability_row[top_index]),
                    probabilities=probabilities,
                    token_count=token_count,
                )
            )

        return results

    def explain(
        self,
        text: str,
        emotion: str,
        max_words: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Estimate word influence using leave-one-word-out probability changes.

        This is an explanation approximation, not a formal attribution method.
        """
        words = text.split()

        if not words:
            return []

        target_index = CLASS_NAMES.index(emotion)
        baseline_probability = self.predict([text])[0].probabilities[emotion]

        scores: list[tuple[str, float]] = []

        for index, word in enumerate(words):
            reduced_words = words[:index] + words[index + 1 :]
            reduced_text = " ".join(reduced_words)

            if not reduced_text.strip():
                reduced_probability = 0.0
            else:
                reduced_probability = self.predict([reduced_text])[0].probabilities[
                    emotion
                ]

            influence = baseline_probability - reduced_probability
            scores.append((word, float(influence)))

        scores.sort(key=lambda item: item[1], reverse=True)

        return scores[:max_words]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Emotion Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --gold: #CDA95B;
    --gold-light: #E3CB8C;
    --gold-deep: #9C7B34;
    --bg: #161615;
    --surface: #1E1D1B;
    --surface-raised: #252422;
    --text: #E4E0D8;
    --muted: #A6A196;
    --line: rgba(255, 255, 255, 0.09);
    --line-soft: rgba(255, 255, 255, 0.06);
    --line-gold: rgba(205, 169, 91, 0.45);
}

html, body, .stApp, .stMarkdown, button, input, textarea, select {
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(22, 22, 21, 0.92);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line-soft);
}

[data-testid="stSidebar"] {
    background: #1A1918;
    border-right: 1px solid var(--line-soft);
}

hr {
    border-color: var(--line-soft) !important;
}

/* Top navigation is fixed-position; leave room so page titles aren't clipped. */
.block-container {
    padding-top: 5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

footer, .stAppDeployButton, [data-testid="stAppDeployButton"] {
    display: none !important;
}

/* The one expressive moment: a gold serif page title. */
.page-title {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif;
    font-size: 2.1rem;
    font-weight: 600;
    letter-spacing: 0;
    line-height: 1.25;
    margin: 0;
    color: var(--gold-light);
}

.page-subtitle {
    max-width: 46rem;
    margin: 0.55rem 0 1.25rem;
    font-size: 1.03rem;
    line-height: 1.65;
    color: var(--muted);
}

.section-title {
    font-size: 1.13rem;
    font-weight: 600;
    margin: 0 0 0.15rem;
    color: var(--text);
}

.section-subtitle {
    font-size: 0.92rem;
    color: var(--muted);
    margin: 0 0 0.9rem;
}

.label {
    font-size: 0.83rem;
    font-weight: 500;
    color: var(--muted);
}

.small {
    font-size: 0.86rem;
    line-height: 1.5;
    color: var(--muted);
}

.panel {
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    background: var(--surface);
}

.result-panel {
    border-radius: 18px;
    padding: 1.6rem 1.7rem;
}

.result-main {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin: 0.45rem 0 0.5rem;
}

.result-emoji {
    font-size: 2.5rem;
    line-height: 1;
}

.result-name {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif;
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1.15;
}

.result-desc {
    font-size: 0.96rem;
    line-height: 1.55;
    opacity: 0.78;
    margin: 0 0 1.35rem;
}

.track {
    height: 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
}

.fill {
    height: 100%;
    border-radius: 999px;
}

.prob-row {
    display: grid;
    grid-template-columns: 8rem 1fr 3.7rem;
    align-items: center;
    gap: 0.9rem;
    padding: 0.5rem 0;
}

.prob-label {
    font-size: 0.94rem;
    opacity: 0.85;
}

.prob-value {
    text-align: right;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
    opacity: 0.85;
}

.prob-row.top .prob-label,
.prob-row.top .prob-value {
    font-weight: 700;
    opacity: 1;
    color: var(--gold-light);
}

.chips {
    line-height: 2.35;
}

.word {
    display: inline-block;
    padding: 0.12rem 0.5rem;
    margin: 0 0.1rem;
    border-radius: 7px;
    font-size: 1rem;
}

.facts {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-top: 1.6rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line-soft);
}

.fact-value {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0.15rem;
}

.note {
    border-left: 3px solid var(--gold-deep);
    padding: 0.65rem 1rem;
    margin-top: 1.25rem;
    font-size: 0.88rem;
    line-height: 1.55;
    opacity: 0.8;
}

.empty-state {
    margin-top: 1.4rem;
    padding: 2.2rem 1.6rem;
    text-align: center;
    border: 1px dashed var(--line);
    border-radius: 16px;
    background: var(--surface);
    font-size: 0.96rem;
    color: var(--muted);
}

.empty-state strong {
    color: var(--gold-light);
}

.footer-note {
    margin-top: 2.5rem;
    text-align: center;
    font-size: 0.8rem;
    line-height: 1.5;
    opacity: 0.55;
}

.char-count {
    text-align: right;
    font-size: 0.83rem;
    opacity: 0.6;
    padding-top: 0.55rem;
}

.spectrum {
    display: flex;
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 1.4rem;
}

.spectrum span {
    flex: 1;
}

.legend-item {
    display: flex;
    gap: 0.7rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--line-soft);
}

.legend-item:last-child {
    border-bottom: 0;
}

.legend-dot {
    flex: none;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 0.42rem;
}

.legend-name {
    font-size: 0.92rem;
    font-weight: 600;
}

/* Bordered containers, metrics, expanders */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 14px;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stExpander"] {
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    background: var(--surface);
}

/* Inputs */
.stTextArea textarea,
.stTextInput input {
    border-radius: 10px;
    font-size: 1rem;
    line-height: 1.55;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--line-soft);
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--gold);
    box-shadow: 0 0 0 1px var(--gold);
}

[data-baseweb="select"] > div {
    background: var(--surface);
    border-color: var(--line-soft);
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px dashed var(--line);
    border-radius: 12px;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1rem;
    background: transparent;
    color: var(--text);
    border: 1px solid var(--line);
}

.stButton button:hover,
.stDownloadButton button:hover {
    border-color: var(--line-gold);
    color: var(--gold-light);
    background: rgba(205, 169, 91, 0.06);
}

.stButton button[kind="primary"],
.stButton button[data-testid="stBaseButton-primary"] {
    background: var(--gold);
    border: 1px solid var(--gold);
    color: #1B1608;
}

.stButton button[kind="primary"]:hover,
.stButton button[data-testid="stBaseButton-primary"]:hover {
    background: var(--gold-light);
    border-color: var(--gold-light);
    color: #1B1608;
}

/* Disabled primary action: visibly inactive instead of a live-looking gold button. */
.stButton button[kind="primary"]:disabled,
.stButton button[data-testid="stBaseButton-primary"]:disabled {
    background: rgba(205, 169, 91, 0.16);
    border-color: transparent;
    color: rgba(228, 224, 216, 0.38);
    cursor: not-allowed;
}

@media (max-width: 760px) {
    .facts {
        grid-template-columns: 1fr;
        gap: 0.9rem;
    }

    .prob-row {
        grid-template-columns: 6rem 1fr 3.4rem;
    }
}
</style>
"""


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def render_html(markup: str) -> None:
    compact = " ".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(compact, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    render_html(
        f"""
        <div class="page-title">{title}</div>
        <p class="page-subtitle">{subtitle}</p>
        """
    )


def section(title: str, subtitle: str = "", top: float = 0) -> None:
    render_html(
        f"""
        <div class="section-title" style="margin-top:{top}rem;">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """
    )


def confidence_band(percentage: float) -> tuple[str, str]:
    if percentage >= 75:
        return "High confidence", "#4CC38A"
    if percentage >= 50:
        return "Moderate confidence", "#E0A93B"
    return "Low confidence", "#E5645B"


def emotion_color(emotion: str) -> str:
    return CLASS_METADATA.get(emotion, {}).get("color", UNCERTAIN_COLOR)


def emotion_emoji(emotion: str) -> str:
    return CLASS_METADATA.get(emotion, {}).get("emoji", "🤔")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    if "input_text" not in st.session_state:
        st.session_state.input_text = next(iter(EXAMPLE_TEXTS.values()))

    if "history" not in st.session_state:
        st.session_state.history = []

    if "last" not in st.session_state:
        st.session_state.last = None

    if "batch_df" not in st.session_state:
        st.session_state.batch_df = None

    if "batch_source" not in st.session_state:
        st.session_state.batch_source = None

    if "selected_row" not in st.session_state:
        st.session_state.selected_row = None

    if "sample_csv_requested" not in st.session_state:
        st.session_state.sample_csv_requested = False


def load_example() -> None:
    st.session_state.input_text = EXAMPLE_TEXTS[st.session_state.example_choice]


def clear_input() -> None:
    st.session_state.input_text = ""
    st.session_state.last = None


def set_example_text(text: str) -> None:
    st.session_state.input_text = text
    st.session_state.last = None


def clear_history() -> None:
    st.session_state.history = []


def use_sample_csv() -> None:
    st.session_state.sample_csv_requested = True


def remove_sample_csv() -> None:
    st.session_state.sample_csv_requested = False


def run_analysis(text: str) -> None:
    started = time.perf_counter()
    result = load_classifier().predict([text])[0]
    latency_ms = (time.perf_counter() - started) * 1000

    st.session_state.last = {
        "result": result,
        "latency_ms": latency_ms,
        "influence": None,
    }

    st.session_state.history.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Text": text,
            "Emotion": result.emotion,
            "Confidence": result.confidence,
        }
    )


def collect_batch_texts(pasted: str, uploaded, use_sample: bool = False) -> list[str]:
    texts = [line.strip() for line in pasted.splitlines() if line.strip()]

    source = uploaded

    if source is None and use_sample:
        source = io.StringIO(SAMPLE_CSV)

    if source is not None:
        try:
            frame = pd.read_csv(source)
        except Exception:
            st.error("That file could not be read. Upload a valid CSV file.")
            return []

        if "text" in frame.columns:
            column = "text"
        else:
            column = frame.columns[0]
            st.info(f"No `text` column found. Using the first column: `{column}`.")

        texts.extend(frame[column].dropna().astype(str).str.strip().tolist())

    texts = [text for text in texts if text]

    if not texts:
        st.warning("Paste some text or upload a CSV file first.")
        return []

    if len(texts) > MAX_BATCH_ROWS:
        st.warning(f"Only the first {MAX_BATCH_ROWS:,} texts were analyzed.")
        texts = texts[:MAX_BATCH_ROWS]

    return texts


def classify_batch(texts: list[str]) -> pd.DataFrame:
    rows = []

    for result in load_classifier().predict(texts):
        row = {
            "text": result.text,
            "emotion": result.emotion,
            "confidence": round(result.confidence, 4),
        }

        row.update(
            {f"p_{name}": round(result.probabilities[name], 4) for name in CLASS_NAMES}
        )

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_batch_frame(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["status"] = np.where(
        result["confidence"] >= threshold,
        result["emotion"],
        "uncertain",
    )
    result["confidence_pct"] = result["confidence"] * 100
    return result


# ---------------------------------------------------------------------------
# Sidebar components
# ---------------------------------------------------------------------------


def render_model_card() -> None:
    render_html(
        f"""
        <div class="panel">
            <div class="label">Model</div>
            <div style="font-size:1.08rem; font-weight:650; margin:0.15rem 0 0.35rem;">
                Bidirectional GRU emotion classifier
            </div>
            <div class="small">
                Predicts six emotions. Text is truncated to the first
                {MAX_SEQUENCE_LENGTH:,} tokens.
            </div>
        </div>
        """
    )


def render_confidence_settings() -> None:
    st.slider(
        "Minimum confidence",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        key="threshold",
        help=(
            "Predictions below this confidence are marked as "
            "“No clear emotion detected”."
        ),
    )

    st.toggle(
        "Explain predictions",
        value=True,
        key="explain",
        help="Highlight words that influenced the prediction.",
    )


def render_example_picker() -> None:
    st.selectbox(
        "Try an example",
        options=list(EXAMPLE_TEXTS.keys()),
        key="example_choice",
        on_change=load_example,
        label_visibility="collapsed",
    )


def render_emotion_legend() -> None:
    legend = "".join(
        f"""
        <div class="legend-item">
            <span class="legend-dot"
                  style="background:{CLASS_METADATA[name]["color"]};">
            </span>
            <div>
                <div class="legend-name">
                    {CLASS_METADATA[name]["emoji"]} {name.title()}
                </div>
                <div class="small">
                    {CLASS_METADATA[name]["description"]}
                </div>
            </div>
        </div>
        """
        for name in CLASS_NAMES
    )

    render_html(f"<div>{legend}</div>")


# ---------------------------------------------------------------------------
# Shared result components
# ---------------------------------------------------------------------------


def render_probability_bars(
    ranked: list[tuple[str, float]],
    selected: str,
) -> None:
    rows = []

    for name, probability in ranked:
        is_selected = name == selected
        width = max(probability * 100, 0.8)

        rows.append(
            f"""
            <div class="prob-row{" top" if is_selected else ""}">
                <div class="prob-label">
                    {emotion_emoji(name)} {name.title()}
                </div>
                <div class="track">
                    <div class="fill"
                         style="width:{width:.1f}%;
                                background:{emotion_color(name)};">
                    </div>
                </div>
                <div class="prob-value">
                    {probability * 100:.1f}%
                </div>
            </div>
            """
        )

    render_html(
        f"""
        <div class="panel">
            <div class="label" style="margin-bottom:0.45rem;">
                How likely each emotion is
            </div>
            {"".join(rows)}
        </div>
        """
    )


def render_word_influence(
    influence: list[tuple[str, float]],
    color: str,
    emotion: str,
) -> None:
    section(
        "Why this result",
        (
            f"Highlighted words increased the likelihood of {emotion.title()}. "
            "Darker highlighting means a stronger influence."
        ),
        top=1.5,
    )

    peak = max(
        (score for _, score in influence if score > 0),
        default=0.0,
    )

    chips = []

    for word, score in influence:
        strength = score / peak if peak > 0 and score > 0 else 0.0

        style = (
            f' style="background:{color}{int(24 + 140 * strength):02x};"'
            if strength > 0
            else ""
        )

        tooltip = f"{score * 100:+.1f} points for {emotion.title()}"

        chips.append(
            f'<span class="word"{style} title="{tooltip}">{escape(word)}</span>'
        )

    with st.container(border=True):
        render_html(f'<div class="chips">{"".join(chips)}</div>')


def render_model_limitations(expanded: bool = False) -> None:
    with st.expander(
        "How to interpret results",
        expanded=expanded,
    ):
        st.markdown(
            """
            ### What this model can do

            Emotion Intelligence estimates which of six emotions best matches a piece
            of text: sadness, joy, love, anger, fear, or surprise.

            ### What it cannot do

            - It does not detect a neutral class.
            - It cannot reliably identify sarcasm, irony, cultural context, or mixed emotions.
            - It should not be used for clinical, psychological, or mental-health diagnosis.
            - Low-confidence predictions should be reviewed by a person before an automated action.

            ### Recommended use

            Use confidence as a routing signal:

            - **High confidence:** suitable for automated sorting or reporting.
            - **Moderate confidence:** review before acting.
            - **Low confidence:** send to human review or treat as inconclusive.
            """
        )


# ---------------------------------------------------------------------------
# Analyze page
# ---------------------------------------------------------------------------


def analyze_page() -> None:
    page_header(
        "Analyze emotional tone",
        (
            "Paste a review, message, support ticket, or comment to understand its "
            "likely emotional tone, confidence, and most influential words."
        ),
    )

    with st.container(border=True):
        user_text = st.text_area(
            "Text to analyze",
            key="input_text",
            height=150,
            max_chars=MAX_INPUT_CHARACTERS,
            label_visibility="collapsed",
            placeholder=("Example: I am really excited about the opportunity..."),
        )

        action_column, clear_column, count_column = st.columns([1.25, 0.8, 3.5])

        analyze_clicked = action_column.button(
            "Analyze text",
            type="primary",
            use_container_width=True,
        )

        clear_column.button(
            "Clear",
            on_click=clear_input,
            use_container_width=True,
        )

        with count_column:
            render_html(
                f"""
                <div class="char-count">
                    {len(user_text):,} / {MAX_INPUT_CHARACTERS:,} characters
                </div>
                """
            )

    st.markdown("##### Quick examples")

    example_columns = st.columns(len(EXAMPLE_TEXTS))

    for column, (label, text) in zip(
        example_columns,
        EXAMPLE_TEXTS.items(),
    ):
        with column:
            st.button(
                label.replace(" statement", ""),
                key=f"example_{label}",
                on_click=set_example_text,
                args=(text,),
                use_container_width=True,
            )

    if analyze_clicked:
        if user_text.strip():
            run_analysis(user_text.strip())
        else:
            st.warning("Enter some text before selecting Analyze text.")

    last = st.session_state.get("last")

    if last is None:
        render_html(
            """
            <div class="empty-state">
                Enter text and select <strong>Analyze text</strong> to see the predicted
                emotion, confidence, probability breakdown, and influential words.
            </div>
            """
        )
        render_model_limitations()
        return

    render_single_result(last)
    render_recent_history()


def render_single_result(last: dict) -> None:
    result = last["result"]
    threshold = st.session_state.threshold
    confident = result.confidence >= threshold
    meta = CLASS_METADATA[result.emotion]
    percentage = result.confidence * 100
    band_label, band_color = confidence_band(percentage)

    if confident:
        accent = meta["color"]
        emoji = meta["emoji"]
        headline = result.emotion.title()
        description = meta["description"]
        panel_style = f"background:{accent}14; border-color:{accent}66;"
        recommendation = (
            "This prediction is above your confidence threshold and can be used "
            "for automated routing or reporting."
        )
    else:
        accent = UNCERTAIN_COLOR
        emoji = "🤔"
        headline = "No clear emotion detected"
        description = (
            f"The closest match is {meta['emoji']} {result.emotion.title()} at "
            f"{percentage:.1f}%, below your minimum threshold of {threshold:.0%}. "
            "Treat this as inconclusive and review it manually if the decision matters."
        )
        panel_style = ""
        recommendation = (
            "Send this item to human review or collect more context before "
            "taking automated action."
        )

    ranked = sorted(
        result.probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    section("Analysis result", top=1.5)

    result_column, probability_column = st.columns(
        [1, 1.25],
        gap="large",
    )

    with result_column:
        render_html(
            f"""
            <div class="panel result-panel" style="{panel_style}">
                <div class="label">Most likely emotion</div>
                <div class="result-main">
                    <span class="result-emoji">{emoji}</span>
                    <span class="result-name">{headline}</span>
                </div>
                <p class="result-desc">{description}</p>

                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="label">Model confidence</span>
                    <span style="font-size:1.55rem; font-weight:700;">
                        {percentage:.1f}%
                    </span>
                </div>

                <div class="track">
                    <div class="fill"
                         style="width:{percentage:.1f}%; background:{accent};">
                    </div>
                </div>

                <span style="
                    display:inline-block;
                    margin-top:0.85rem;
                    padding:0.18rem 0.65rem;
                    border-radius:999px;
                    font-size:0.8rem;
                    font-weight:600;
                    background:{band_color}22;
                    color:{band_color};
                    border:1px solid {band_color}55;
                ">
                    {band_label}
                </span>
            </div>
            """
        )

    with probability_column:
        render_probability_bars(ranked, result.emotion)

    runner_name, runner_probability = ranked[1]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Second-most likely",
        runner_name.title(),
        f"{runner_probability * 100:.1f}%",
    )

    metric_2.metric(
        "Words considered",
        f"{min(result.token_count, MAX_SEQUENCE_LENGTH):,}",
        f"of {MAX_SEQUENCE_LENGTH:,}",
    )

    metric_3.metric(
        "Inference time",
        f"{last['latency_ms']:.0f} ms",
    )

    metric_4.metric(
        "Recommended action",
        "Use prediction" if confident else "Review manually",
    )

    st.info(recommendation, icon="💡")

    if st.session_state.explain:
        if last["influence"] is None:
            with st.spinner("Finding influential words..."):
                last["influence"] = load_classifier().explain(
                    result.text,
                    result.emotion,
                )

        render_word_influence(
            last["influence"],
            meta["color"],
            result.emotion,
        )

    with st.expander("View detailed probabilities"):
        probability_table = pd.DataFrame(
            {
                "Emotion": [name.title() for name, _ in ranked],
                "Probability": [f"{probability:.4f}" for _, probability in ranked],
            }
        )

        st.dataframe(
            probability_table,
            use_container_width=True,
            hide_index=True,
        )

    render_model_limitations()


def render_recent_history() -> None:
    history = st.session_state.history

    if not history:
        return

    threshold = st.session_state.threshold
    frame = pd.DataFrame(history[::-1])

    summary = pd.DataFrame(
        {
            "Time": frame["Time"],
            "Text": frame["Text"],
            "Result": np.where(
                frame["Confidence"] >= threshold,
                frame["Emotion"].str.title(),
                "No clear emotion detected",
            ),
            "Confidence": ((frame["Confidence"] * 100).round(1).astype(str) + "%"),
        }
    )

    section(
        "Recent analyses",
        "Your latest analyses in this browser session.",
        top=2,
    )

    st.dataframe(
        summary.head(10),
        use_container_width=True,
        hide_index=True,
    )

    download_column, clear_column = st.columns(2)

    download_column.download_button(
        "Download history",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="emotion_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    clear_column.button(
        "Clear history",
        on_click=clear_history,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Batch analysis page
# ---------------------------------------------------------------------------


def batch_page() -> None:
    page_header(
        "Batch emotion analysis",
        (
            "Upload a CSV or paste multiple texts to classify them in one run, "
            "inspect confidence, filter results, and export the output."
        ),
    )

    with st.container(border=True):
        input_method = st.radio(
            "Choose input method",
            options=["Upload CSV", "Paste texts"],
            horizontal=True,
            key="batch_input_method",
        )

        uploaded = None
        pasted = ""
        sample_active = False

        if input_method == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help=(
                    "The app looks for a column named `text`. If it is absent, "
                    "the first column is used."
                ),
            )

            if uploaded is not None:
                # An uploaded file always takes priority over the sample.
                st.session_state.sample_csv_requested = False
                st.success(f"Uploaded: **{uploaded.name}**")

            sample_active = uploaded is None and st.session_state.sample_csv_requested

            sample_column, download_column, _ = st.columns([1.2, 1.5, 2.5])

            sample_column.button(
                "Use sample CSV",
                on_click=use_sample_csv,
                disabled=uploaded is not None,
                use_container_width=True,
                help="Load a 7-row example so you can try the analysis.",
            )

            download_column.download_button(
                "Download sample CSV",
                data=SAMPLE_CSV.encode("utf-8"),
                file_name="emotion_sample.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if sample_active:
                info_column, remove_column = st.columns([4, 1])

                info_column.info(
                    "Sample CSV loaded (7 texts). Select **Analyze all** to run it."
                )

                remove_column.button(
                    "Remove",
                    on_click=remove_sample_csv,
                    use_container_width=True,
                )

        else:
            pasted = st.text_area(
                "Texts to analyze",
                key="batch_text",
                height=180,
                label_visibility="collapsed",
                placeholder="Enter one text per line...",
            )

        if input_method == "Upload CSV":
            ready = uploaded is not None or sample_active
            disabled_hint = "Upload a CSV or load the sample CSV first."
        else:
            ready = bool(pasted.strip())
            disabled_hint = "Enter at least one text first."

        analyze_clicked = st.button(
            "Analyze all",
            type="primary",
            use_container_width=True,
            disabled=not ready,
            help=None if ready else disabled_hint,
        )

    if analyze_clicked:
        texts = collect_batch_texts(pasted, uploaded, sample_active)

        if texts:
            with st.spinner(f"Analyzing {len(texts):,} texts..."):
                st.session_state.batch_df = classify_batch(texts)

                if uploaded is not None:
                    source_name = uploaded.name
                elif sample_active:
                    source_name = "Sample CSV"
                else:
                    source_name = "Pasted text"

                st.session_state.batch_source = source_name
                st.session_state.selected_row = None

    frame = st.session_state.get("batch_df")

    if frame is None:
        render_html(
            """
            <div class="empty-state">
                Upload a CSV or paste texts, then select
                <strong>Analyze all</strong> to see the emotion distribution,
                confidence breakdown, and downloadable results.
            </div>
            """
        )
        render_model_limitations()
        return

    threshold = st.session_state.threshold
    frame = prepare_batch_frame(frame, threshold)

    render_batch_summary(frame, threshold)
    render_batch_filters_and_results(frame, threshold)


def render_batch_summary(
    frame: pd.DataFrame,
    threshold: float,
) -> None:
    total = len(frame)
    confident_count = int((frame["confidence"] >= threshold).sum())
    uncertain_count = total - confident_count

    emotion_counts = frame["emotion"].value_counts()
    dominant_emotion = emotion_counts.index[0] if not emotion_counts.empty else "—"
    average_confidence = frame["confidence"].mean() * 100

    section(
        f"Results for {total:,} texts",
        f"Source: {st.session_state.get('batch_source', 'Unknown')}",
        top=1.5,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Texts analyzed",
        f"{total:,}",
    )

    metric_2.metric(
        "High-confidence",
        f"{confident_count:,}",
        f"{confident_count / total * 100:.1f}%",
    )

    metric_3.metric(
        "Needs review",
        f"{uncertain_count:,}",
        f"{uncertain_count / total * 100:.1f}%",
    )

    metric_4.metric(
        "Average confidence",
        f"{average_confidence:.1f}%",
    )

    chart_column, distribution_column = st.columns(
        [1.15, 1],
        gap="large",
    )

    with chart_column:
        st.markdown("##### Emotion distribution")

        chart_data = pd.DataFrame(
            {
                "Emotion": [name.title() for name in CLASS_NAMES],
                "Texts": [int(emotion_counts.get(name, 0)) for name in CLASS_NAMES],
            }
        )

        st.bar_chart(
            chart_data,
            x="Emotion",
            y="Texts",
            color=PRIMARY_COLOR,
            use_container_width=True,
        )

    with distribution_column:
        rows = []

        for name in [*CLASS_NAMES, "uncertain"]:
            count = int((frame["status"] == name).sum())

            if name == "uncertain" and count == 0:
                continue

            color = UNCERTAIN_COLOR if name == "uncertain" else emotion_color(name)
            emoji = "🤔" if name == "uncertain" else emotion_emoji(name)
            title = "No clear emotion" if name == "uncertain" else name.title()
            width = max(count / total * 100, 0.8)

            rows.append(
                f"""
                <div class="prob-row">
                    <div class="prob-label">{emoji} {title}</div>
                    <div class="track">
                        <div class="fill"
                             style="width:{width:.1f}%; background:{color};">
                        </div>
                    </div>
                    <div class="prob-value">{count:,}</div>
                </div>
                """
            )

        render_html(
            f"""
            <div class="panel">
                <div class="label" style="margin-bottom:0.45rem;">
                    Classification breakdown
                </div>
                {"".join(rows)}
            </div>
            """
        )


def render_batch_filters_and_results(
    frame: pd.DataFrame,
    threshold: float,
) -> None:
    section(
        "Explore results",
        "Filter predictions, inspect individual texts, and download the results.",
        top=1.75,
    )

    filter_column_1, filter_column_2, filter_column_3 = st.columns([1.1, 1.1, 2])

    with filter_column_1:
        emotion_filter = st.selectbox(
            "Emotion",
            options=[
                "All emotions",
                *[name.title() for name in CLASS_NAMES],
                "No clear emotion detected",
            ],
            key="batch_emotion_filter",
        )

    with filter_column_2:
        confidence_filter = st.selectbox(
            "Confidence",
            options=[
                "All confidence",
                "High confidence",
                "Moderate confidence",
                "Low confidence",
            ],
            key="batch_confidence_filter",
        )

    with filter_column_3:
        search_term = st.text_input(
            "Search text",
            placeholder="Search for a word or phrase...",
            key="batch_search",
        )

    filtered = frame.copy()

    if emotion_filter == "No clear emotion detected":
        filtered = filtered[filtered["status"] == "uncertain"]
    elif emotion_filter != "All emotions":
        filtered = filtered[filtered["emotion"] == emotion_filter.lower()]

    if confidence_filter == "High confidence":
        filtered = filtered[filtered["confidence"] >= 0.75]
    elif confidence_filter == "Moderate confidence":
        filtered = filtered[
            (filtered["confidence"] >= 0.50) & (filtered["confidence"] < 0.75)
        ]
    elif confidence_filter == "Low confidence":
        filtered = filtered[filtered["confidence"] < 0.50]

    if search_term.strip():
        filtered = filtered[
            filtered["text"].str.contains(
                search_term.strip(),
                case=False,
                na=False,
            )
        ]

    if filtered.empty:
        st.warning("No results match the selected filters.")
        return

    display = pd.DataFrame(
        {
            "Text": filtered["text"],
            "Predicted emotion": filtered["emotion"].str.title(),
            "Confidence": filtered["confidence_pct"].round(1).astype(str) + "%",
            "Status": filtered["status"].map(
                {
                    **{name: name.title() for name in CLASS_NAMES},
                    "uncertain": "No clear emotion detected",
                }
            ),
        }
    )

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selection = event.selection.rows

    if selection:
        selected_index = filtered.index[selection[0]]
        selected_result = filtered.loc[selected_index]

        with st.container(border=True):
            st.markdown("##### Selected prediction")

            selected_column, detail_column = st.columns(
                [1, 1.4],
                gap="large",
            )

            with selected_column:
                emotion = selected_result["emotion"]
                confidence = selected_result["confidence"] * 100
                band_label, _ = confidence_band(confidence)

                st.metric(
                    "Predicted emotion",
                    emotion.title(),
                    f"{confidence:.1f}% confidence",
                )

                st.progress(
                    min(confidence / 100, 1.0),
                    text=band_label,
                )

                st.caption(selected_result["text"])

            with detail_column:
                probabilities = [
                    (name, selected_result[f"p_{name}"]) for name in CLASS_NAMES
                ]
                probabilities.sort(
                    key=lambda item: item[1],
                    reverse=True,
                )

                render_probability_bars(
                    probabilities,
                    selected_result["emotion"],
                )

    export_column_1, export_column_2 = st.columns(2)

    export_column_1.download_button(
        "Download filtered results",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="emotion_batch_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    export_column_2.download_button(
        "Download all results",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="emotion_batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    render_model_limitations()


# ---------------------------------------------------------------------------
# Insights page
# ---------------------------------------------------------------------------


def insights_page() -> None:
    page_header(
        "Insights",
        (
            "Understand patterns across a completed batch analysis: emotion mix, "
            "confidence quality, and examples that may need human review."
        ),
    )

    frame = st.session_state.get("batch_df")

    if frame is None:
        render_html(
            """
            <div class="empty-state">
                Run a batch analysis first to generate insights.
                Go to <strong>Batch analysis</strong>, upload a CSV or paste texts,
                and select <strong>Analyze all</strong>.
            </div>
            """
        )
        return

    threshold = st.session_state.threshold
    frame = prepare_batch_frame(frame, threshold)
    total = len(frame)

    emotion_counts = frame["emotion"].value_counts()
    uncertain_count = int((frame["status"] == "uncertain").sum())
    average_confidence = frame["confidence"].mean() * 100

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Dominant emotion",
        emotion_counts.index[0].title() if not emotion_counts.empty else "—",
    )

    metric_2.metric(
        "Average confidence",
        f"{average_confidence:.1f}%",
    )

    metric_3.metric(
        "Low-confidence items",
        f"{uncertain_count:,}",
        f"{uncertain_count / total * 100:.1f}%",
    )

    metric_4.metric(
        "Dataset size",
        f"{total:,}",
    )

    chart_column, quality_column = st.columns(
        [1.2, 1],
        gap="large",
    )

    with chart_column:
        st.markdown("##### Emotion mix")

        chart_data = pd.DataFrame(
            {
                "Emotion": [name.title() for name in CLASS_NAMES],
                "Share": [
                    emotion_counts.get(name, 0) / total * 100 for name in CLASS_NAMES
                ],
            }
        )

        st.bar_chart(
            chart_data,
            x="Emotion",
            y="Share",
            color=PRIMARY_COLOR,
            use_container_width=True,
        )

    with quality_column:
        st.markdown("##### Confidence quality")

        high = int((frame["confidence"] >= 0.75).sum())
        moderate = int(
            ((frame["confidence"] >= 0.50) & (frame["confidence"] < 0.75)).sum()
        )
        low = int((frame["confidence"] < 0.50).sum())

        quality = pd.DataFrame(
            {
                "Confidence band": [
                    "High",
                    "Moderate",
                    "Low",
                ],
                "Texts": [high, moderate, low],
            }
        )

        st.bar_chart(
            quality,
            x="Confidence band",
            y="Texts",
            color="#E3CB8C",
            use_container_width=True,
        )

    review_section, examples_section = st.columns(
        [1, 1.4],
        gap="large",
    )

    with review_section:
        st.markdown("##### Items to review")

        review = frame[frame["confidence"] < threshold].copy()

        if review.empty:
            st.success("No items fall below the current confidence threshold.")
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Text": review["text"],
                        "Closest emotion": review["emotion"].str.title(),
                        "Confidence": review["confidence_pct"].round(1).astype(str)
                        + "%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with examples_section:
        st.markdown("##### Example predictions")

        example_columns = st.columns(2)

        for column, emotion in zip(
            example_columns,
            CLASS_NAMES,
        ):
            with column:
                subset = frame[frame["emotion"] == emotion]

                if subset.empty:
                    st.caption(f"No {emotion.title()} examples.")
                    continue

                example = subset.sort_values(
                    "confidence",
                    ascending=False,
                ).iloc[0]

                st.markdown(
                    f"""
                    **{emotion_emoji(emotion)} {emotion.title()}**  
                    {example["text"][:180]}
                    """
                )
                st.caption(f"Confidence: {example['confidence'] * 100:.1f}%")

    render_model_limitations(expanded=True)


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------


def about_page() -> None:
    page_header(
        "About Emotion Intelligence",
        (
            "Understand what the platform predicts, how confidence should be used, "
            "and where human review remains essential."
        ),
    )

    overview_column, model_column = st.columns(
        [1.2, 1],
        gap="large",
    )

    with overview_column:
        st.markdown("### What this platform does")

        st.markdown(
            """
            Emotion Intelligence helps you understand the emotional tone of written
            text. It is designed for exploratory analysis, customer-feedback triage,
            content review, and internal research workflows.

            You can:

            - Analyze one piece of text and inspect influential words.
            - Analyze many texts from a CSV or pasted input.
            - Set a confidence threshold for uncertain predictions.
            - Filter and export batch results.
            - Review dataset-level patterns in the Insights page.
            """
        )

    with model_column:
        render_model_card()

        st.markdown("### Supported emotions")
        render_emotion_legend()

    render_model_limitations(expanded=True)

    st.markdown("### Responsible use")

    st.warning(
        """
        Emotion predictions are probabilistic estimates, not facts about a person’s
        internal state. Do not use this platform for mental-health diagnosis,
        employment decisions, disciplinary action, medical care, or other
        high-stakes decisions without appropriate human oversight.
        """,
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧠 Emotion Intelligence")
        st.caption("Understand emotional tone, confidence, and influential words.")

        st.divider()
        render_confidence_settings()

        st.divider()
        render_example_picker()

        st.divider()
        st.markdown("##### Supported emotions")
        render_emotion_legend()

        st.divider()
        st.button(
            "Clear session history",
            on_click=clear_history,
            use_container_width=True,
        )

    try:
        load_classifier()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    # Initialize Session State before pages access it.
    init_state()

    pages = [
        st.Page(
            analyze_page,
            title="Analyze text",
            icon=":material/edit_note:",
            default=True,
        ),
        st.Page(
            batch_page,
            title="Batch analysis",
            icon=":material/upload_file:",
        ),
        st.Page(
            insights_page,
            title="Insights",
            icon=":material/insights:",
        ),
        st.Page(
            about_page,
            title="About & limitations",
            icon=":material/info:",
        ),
    ]

    page = st.navigation(pages, position="top")
    page.run()

    render_html(
        """
        <div class="footer-note">
            Emotion Intelligence indicates likely emotional tone only.
            It is not intended for clinical, psychological, or mental-health diagnosis.
        </div>
        """
    )


if __name__ == "__main__":
    main()
