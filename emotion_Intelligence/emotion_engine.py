"""
Emotion Engine
==============

A small, reusable wrapper around the trained Bidirectional GRU model. Copy this
file and the `artifacts/` folder into any backend to classify text:

    from emotion_engine import EmotionClassifier

    classifier = EmotionClassifier("artifacts")
    result = classifier.predict(["i feel so loved and cared for by my family"])[0]
    print(result.emotion, result.confidence)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Must match the label order used by main.py during training.
CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MAX_SEQUENCE_LENGTH = 50


@dataclass(frozen=True)
class Result:
    """Prediction for one text."""

    text: str
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    token_count: int


class EmotionClassifier:
    """Loads the model and tokenizer once and classifies batches of text."""

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        directory = Path(artifacts_dir)
        model_path = directory / "bidirectional_gru_model.keras"
        tokenizer_path = directory / "tokenizer.pkl"

        for path in (model_path, tokenizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact not found: `{path}`. Run `python main.py` first to "
                    "train the model and export the artifacts."
                )

        self.model = load_model(model_path)
        with tokenizer_path.open("rb") as file:
            self.tokenizer = pickle.load(file)

    def _encode(self, texts: list[str]) -> tuple[np.ndarray, list[int]]:
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )
        return padded, [len(sequence) for sequence in sequences]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), 6) with class probabilities."""
        padded, _ = self._encode(texts)
        return self.model.predict(padded, verbose=0, batch_size=256)

    def predict(self, texts: list[str]) -> list[Result]:
        """Classify each text and return one Result per input."""
        texts = [str(text) for text in texts]
        if not texts:
            return []

        padded, token_counts = self._encode(texts)
        probabilities = self.model.predict(padded, verbose=0, batch_size=256)

        results = []
        for text, row, count in zip(texts, probabilities, token_counts):
            index = int(np.argmax(row))
            results.append(
                Result(
                    text=text,
                    emotion=CLASS_NAMES[index],
                    confidence=float(row[index]),
                    probabilities={n: float(p) for n, p in zip(CLASS_NAMES, row)},
                    token_count=count,
                )
            )
        return results

    def explain(
        self,
        text: str,
        emotion: str | None = None,
        max_words: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Score how much each word supports an emotion.

        Each word is removed in turn and the drop in the emotion's probability
        is its score. Positive scores mean the word pushed the model toward
        that emotion. Defaults to the predicted emotion.
        """
        words = text.split()
        if len(words) < 2:
            return [(word, 0.0) for word in words]

        base = self.predict_proba([text])[0]
        target = CLASS_NAMES.index(emotion) if emotion else int(np.argmax(base))

        limit = min(len(words), max_words)
        variants = [" ".join(words[:i] + words[i + 1 :]) for i in range(limit)]
        scores = base[target] - self.predict_proba(variants)[:, target]

        influence = [(word, float(score)) for word, score in zip(words[:limit], scores)]
        influence += [(word, 0.0) for word in words[limit:]]
        return influence
