"""
Evaluate the exported Bidirectional GRU without retraining.

Run from the project directory:
    python evaluate_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Matches the directory used by your training script when run
# from the project directory.
ARTIFACTS_DIR = Path("artifacts")

MODEL_PATH = ARTIFACTS_DIR / "bidirectional_gru_model.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"

CLASS_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

MAX_SEQUENCE_LENGTH = 50


def main() -> None:
    for path in (MODEL_PATH, TOKENIZER_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing artifact: {path.resolve()}\n"
                "Run this script from the directory containing "
                "your artifacts folder."
            )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    # Load only your own trusted tokenizer artifact.
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    test_dataset = load_dataset(
        "dair-ai/emotion",
        split="test",
    )

    actual_class_names = test_dataset.features["label"].names

    if actual_class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {actual_class_names}")

    sequences = tokenizer.texts_to_sequences(list(test_dataset["text"]))

    x_test = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    y_test = np.asarray(
        test_dataset["label"],
        dtype=np.int64,
    )

    probabilities = model.predict(
        x_test,
        batch_size=32,
        verbose=1,
    )

    if probabilities.shape != (len(y_test), len(CLASS_NAMES)):
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    print("\nHELD-OUT TEST RESULTS")
    print(f"Test examples: {len(y_test):,}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nPER-CLASS REPORT")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Bidirectional GRU",
                "test_examples": len(y_test),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        ]
    )

    metrics.to_csv(
        ARTIFACTS_DIR / "bigru_test_metrics.csv",
        index=False,
    )

    print("\nRESUME STATEMENT")
    print(
        f"Achieved {accuracy:.2%} test accuracy and "
        f"{macro_f1:.4f} macro-F1 on the held-out test set."
    )


if __name__ == "__main__":
    main()
