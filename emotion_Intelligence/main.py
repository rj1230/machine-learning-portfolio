"""
Emotion Classification - Training Pipeline
==========================================

Trains and evaluates RNN, LSTM, GRU, and Bidirectional GRU models on the
Hugging Face `dair-ai/emotion` dataset, then exports the Bidirectional GRU
model and tokenizer for the Streamlit app (app.py).

Run:
    python main.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
    SimpleRNN,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "dair-ai/emotion"

# app.py maps predictions to names in this order, so training must match it.
EXPECTED_CLASS_ORDER = ["sadness", "joy", "love", "anger", "fear", "surprise"]

MAX_VOCABULARY_SIZE = 10_000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIMENSION = 128
ADVANCED_EMBEDDING_DIMENSION = 300
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
DROPOUT_RATE = 0.5

# 0 = no class weighting, 1 = fully "balanced" weights. Fully balanced weights
# make the model over-predict rare classes (anger, fear) on ambiguous text, so a
# softened value is used.
CLASS_WEIGHT_POWER = 0.5

# Set to False to train only the Bidirectional GRU that the app uses (faster).
RUN_BASELINE_MODELS = True

RANDOM_SEED = 42
ARTIFACTS_DIR = Path("artifacts")

# (text, expected emotion) pairs checked after training, including short inputs.
SANITY_CHECKS = [
    ("I can't believe how happy I am right now, this is amazing!", "joy"),
    ("I feel so alone and hopeless today.", "sadness"),
    ("I am furious that they cancelled the trip at the last minute.", "anger"),
    ("I feel terrified when walking down dark alleyways alone.", "fear"),
    ("I was shocked and completely surprised by the unexpected gift!", "surprise"),
    ("I really appreciate everything you do for me and care about you deeply.", "love"),
    ("i am happy today", "joy"),
    ("i feel happy", "joy"),
    ("i am so angry", "anger"),
    ("i am scared", "fear"),
    ("i am sad", "sadness"),
    ("i love you", "love"),
]


class Splits(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def load_emotion_dataset():
    """Load the emotion dataset and verify its class order matches the app."""
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    class_names = dataset["train"].features["label"].names

    if class_names != EXPECTED_CLASS_ORDER:
        raise ValueError(
            f"Dataset class order {class_names} does not match the order app.py "
            f"expects {EXPECTED_CLASS_ORDER}. Update CLASS_NAMES in app.py."
        )

    return dataset, class_names


def explore_training_data(dataset, class_names) -> None:
    """Display data-quality checks and save the class-distribution chart."""
    train_df = pd.DataFrame(
        {
            "text": dataset["train"]["text"],
            "label": [class_names[label] for label in dataset["train"]["label"]],
        }
    )

    print("Missing values per column:")
    print(train_df.isnull().sum())

    print("\nClass distribution:")
    print(train_df["label"].value_counts())

    word_counts = train_df["text"].str.split().str.len()
    print(
        f"\nWords per text: mean {word_counts.mean():.1f}, median {word_counts.median():.0f}"
    )

    plt.figure(figsize=(10, 5))
    sns.countplot(
        x="label",
        data=train_df,
        order=train_df["label"].value_counts().index,
    )
    plt.title("Training Set Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("emotion_class_distribution.png", dpi=300)
    plt.close()

    print("Saved class-distribution chart: emotion_class_distribution.png")


def encode_texts(tokenizer: Tokenizer, texts) -> np.ndarray:
    """Convert texts to padded token sequences (same settings as app.py)."""
    sequences = tokenizer.texts_to_sequences(list(texts))
    return pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )


def prepare_text_sequences(dataset):
    """Fit the tokenizer on training text and encode all three splits."""
    tokenizer = Tokenizer(num_words=MAX_VOCABULARY_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(dataset["train"]["text"])

    splits = Splits(
        x_train=encode_texts(tokenizer, dataset["train"]["text"]),
        y_train=np.array(dataset["train"]["label"]),
        x_val=encode_texts(tokenizer, dataset["validation"]["text"]),
        y_val=np.array(dataset["validation"]["label"]),
        x_test=encode_texts(tokenizer, dataset["test"]["text"]),
        y_test=np.array(dataset["test"]["label"]),
    )

    num_classes = len(np.unique(splits.y_train))

    print(f"Vocabulary size: {len(tokenizer.word_index):,}")
    print(f"Maximum sequence length: {splits.x_train.shape[1]}")
    print(f"Number of classes: {num_classes}")
    print(
        f"Split sizes: train {len(splits.y_train):,}, "
        f"validation {len(splits.y_val):,}, test {len(splits.y_test):,}"
    )

    return tokenizer, splits, num_classes


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Calculate softened class weights for the imbalanced emotion classes."""
    class_labels = np.unique(y_train)
    balanced = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=y_train,
    )
    softened = balanced**CLASS_WEIGHT_POWER

    return {int(label): float(weight) for label, weight in zip(class_labels, softened)}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
# mask_zero=True is essential: sequences are post-padded with zeros, and without
# a mask the recurrent layers read dozens of padding steps after short texts,
# which washes out the real words and produces unreliable predictions.


def build_recurrent_model(recurrent_layer, num_classes, model_name):
    """Build a two-layer recurrent neural network."""
    model = Sequential(
        [
            Input(shape=(MAX_SEQUENCE_LENGTH,)),
            Embedding(
                input_dim=MAX_VOCABULARY_SIZE,
                output_dim=EMBEDDING_DIMENSION,
                mask_zero=True,
            ),
            recurrent_layer(128, return_sequences=True),
            Dropout(DROPOUT_RATE),
            recurrent_layer(64),
            Dropout(DROPOUT_RATE),
            Dense(num_classes, activation="softmax"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_bidirectional_gru_model(num_classes):
    """Build the advanced Bidirectional GRU model."""
    model = Sequential(
        [
            Input(shape=(MAX_SEQUENCE_LENGTH,)),
            Embedding(
                input_dim=MAX_VOCABULARY_SIZE,
                output_dim=ADVANCED_EMBEDDING_DIMENSION,
                mask_zero=True,
            ),
            Bidirectional(GRU(128, return_sequences=True)),
            Dropout(DROPOUT_RATE),
            Bidirectional(GRU(64)),
            Dropout(DROPOUT_RATE),
            Dense(num_classes, activation="softmax"),
        ],
        name="bidirectional_gru",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_early_stopping_callback():
    """Create an early-stopping callback."""
    return EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate_model(model, model_name, splits: Splits, class_weights):
    """Train on the train split, stop on validation, and score on the test split."""
    print(f"\n{'=' * 70}")
    print(f"Training model: {model_name}")
    print(f"{'=' * 70}")

    model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(splits.x_val, splits.y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[create_early_stopping_callback()],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(splits.x_test, splits.y_test, verbose=0)

    print(f"{model_name} test loss: {test_loss:.4f}")
    print(f"{model_name} test accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy


# ---------------------------------------------------------------------------
# Evaluation and export
# ---------------------------------------------------------------------------


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save the confusion matrix for the Bidirectional GRU model."""
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Bidirectional GRU")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    plt.savefig("bigru_confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved confusion matrix: bigru_confusion_matrix.png")


def report_test_metrics(y_true, y_pred, x_test, class_names):
    """Print per-class metrics and accuracy by text length."""
    print("\nClassification report (Bidirectional GRU):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    token_counts = (x_test != 0).sum(axis=1)
    buckets = pd.cut(
        token_counts,
        bins=[0, 5, 10, 20, 30, MAX_SEQUENCE_LENGTH],
        labels=["1-5", "6-10", "11-20", "21-30", "31+"],
    )
    by_length = (
        pd.DataFrame({"tokens": buckets, "correct": y_true == y_pred})
        .groupby("tokens", observed=True)["correct"]
        .agg(["mean", "count"])
    )
    by_length.columns = ["Accuracy", "Samples"]

    print("Accuracy by text length (tokens):")
    print(by_length.round(3))


def run_sanity_checks(model, tokenizer, class_names):
    """Classify known sentences, including very short ones, and report misses."""
    texts = [text for text, _ in SANITY_CHECKS]
    probabilities = model.predict(encode_texts(tokenizer, texts), verbose=0)

    print("\nSanity checks:")
    passed = 0
    for (text, expected), row in zip(SANITY_CHECKS, probabilities):
        predicted = class_names[int(np.argmax(row))]
        matched = predicted == expected
        passed += int(matched)
        status = "PASS" if matched else "FAIL"
        print(
            f"[{status}] {text!r} -> {predicted} ({row.max():.1%}), expected {expected}"
        )

    print(f"{passed}/{len(SANITY_CHECKS)} sanity checks passed")


def save_artifacts(model, tokenizer):
    """Save the trained model and tokenizer."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "bidirectional_gru_model.keras"
    tokenizer_path = ARTIFACTS_DIR / "tokenizer.pkl"

    model.save(model_path)

    with tokenizer_path.open("wb") as file:
        pickle.dump(tokenizer, file)

    print(f"\nSaved model: {model_path}")
    print(f"Saved tokenizer: {tokenizer_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete training and evaluation workflow."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    dataset, class_names = load_emotion_dataset()
    explore_training_data(dataset, class_names)

    tokenizer, splits, num_classes = prepare_text_sequences(dataset)

    class_weights = compute_class_weights(splits.y_train)
    print(
        f"Class weights: { {class_names[k]: round(v, 2) for k, v in class_weights.items()} }"
    )

    results = []

    if RUN_BASELINE_MODELS:
        baseline_builders = {
            "Simple RNN": lambda: build_recurrent_model(
                SimpleRNN, num_classes, "simple_rnn"
            ),
            "LSTM": lambda: build_recurrent_model(LSTM, num_classes, "lstm"),
            "GRU": lambda: build_recurrent_model(GRU, num_classes, "gru"),
        }

        for model_name, build_model in baseline_builders.items():
            test_loss, test_accuracy = train_and_evaluate_model(
                build_model(), model_name, splits, class_weights
            )
            results.append(
                {
                    "Model": model_name,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_accuracy,
                }
            )

        print("\nBaseline model comparison:")
        print(pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False))

    print("\nTraining advanced Bidirectional GRU model...")
    bigru_model = build_bidirectional_gru_model(num_classes)
    bigru_model.summary()

    bigru_loss, bigru_accuracy = train_and_evaluate_model(
        bigru_model, "Bidirectional GRU", splits, class_weights
    )
    results.append(
        {
            "Model": "Bidirectional GRU",
            "Test Loss": bigru_loss,
            "Test Accuracy": bigru_accuracy,
        }
    )

    print("\nFinal model comparison:")
    print(pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False))

    bigru_predictions = np.argmax(bigru_model.predict(splits.x_test, verbose=0), axis=1)

    plot_confusion_matrix(splits.y_test, bigru_predictions, class_names)
    report_test_metrics(splits.y_test, bigru_predictions, splits.x_test, class_names)
    run_sanity_checks(bigru_model, tokenizer, class_names)

    save_artifacts(bigru_model, tokenizer)


if __name__ == "__main__":
    main()
