#!/usr/bin/env python3
"""
Heart Disease Prediction - reproducible training and inference pipeline.

Dataset : UCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach).
Target  : `num` (0-4 angiographic severity), converted to binary: num > 0 -> disease.

Improvements in this version
----------------------------
- Data-quality audit: missing values, mixed-type fields, invalid values, duplicates.
- Optional removal of duplicate feature rows before splitting.
- Optional XGBoost model, automatically included when xgboost is installed.
- Calibration, PR-AUC, Brier score, confusion matrix, and bootstrap intervals.
- Permutation importance on the held-out test set.
- Leave-one-site-out validation.
- Optional MLflow experiment tracking.
- Model card generation for reproducibility and responsible-use documentation.

Usage (Windows PowerShell)
--------------------------
    python main.py train --data heart_disease.csv --n-iter 10 --n-jobs 1
    python main.py train --data heart_disease.csv --drop-duplicate-rows
    python main.py predict --input patients.csv --output predictions.csv

Optional dependencies:
    pip install xgboost mlflow

This project is a learning/portfolio project and is not a medical device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Limit BLAS / OpenMP thread pools BEFORE numpy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

LOGGER = logging.getLogger("heart_disease")

TARGET_COLUMN = "num"
SITE_COLUMN = "dataset"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
BINARY_FEATURES = ["fbs", "exang"]
CATEGORICAL_FEATURES = ["sex", "cp", "restecg", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 110),
    "trestbps": (50, 260),
    "chol": (80, 700),
    "thalch": (50, 250),
    "oldpeak": (-3.5, 10),
    "ca": (0, 4),
}

_MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "?", "<na>"}
_TRUE_TOKENS = {"true", "1", "1.0", "yes", "y", "t"}
_FALSE_TOKENS = {"false", "0", "0.0", "no", "n", "f"}


class DataValidationError(ValueError):
    """Raised when input data does not match the expected schema."""


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    output_dir: Path = Path("artifacts")
    test_size: float = 0.2
    cv_folds: int = 5
    n_iter: int = 20
    min_recall: float = 0.90
    include_site: bool = False
    cross_site: bool = True
    n_bootstrap: int = 1000
    seed: int = 42
    n_jobs: int = 1
    drop_duplicate_rows: bool = False
    track_mlflow: bool = False


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def configure_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# --------------------------------------------------------------------------- #
# Data loading, validation and cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, na_values=["?"])
    LOGGER.info("Loaded %s rows x %s columns from %s", len(df), df.shape[1], path)
    return df


def _normalise_text(series: pd.Series) -> pd.Series:
    def clean(value: Any) -> Any:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        return np.nan if text in _MISSING_TOKENS else text

    return series.map(clean).astype("object")


def _to_binary(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> float:
        if value in _TRUE_TOKENS:
            return 1.0
        if value in _FALSE_TOKENS:
            return 0.0
        return np.nan

    return _normalise_text(series).map(convert).astype("float64")


def build_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COLUMN}' not found.")

    raw = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if raw.isna().any():
        raise DataValidationError(
            f"{int(raw.isna().sum())} row(s) have a missing or non-numeric '{TARGET_COLUMN}'."
        )

    if not raw.isin([0, 1, 2, 3, 4]).all():
        raise DataValidationError(
            f"'{TARGET_COLUMN}' must only contain 0, 1, 2, 3 or 4."
        )

    return (raw > 0).astype(int).rename("target")


def prepare_features(
    df: pd.DataFrame, *, include_site: bool, strict: bool
) -> pd.DataFrame:
    wanted = ALL_FEATURES + ([SITE_COLUMN] if include_site else [])
    absent = [column for column in wanted if column not in df.columns]

    if absent:
        if strict:
            raise DataValidationError(f"Missing required column(s): {absent}")
        LOGGER.warning(
            "Input is missing column(s) %s - treated as missing values.", absent
        )

    nan_column = pd.Series(np.nan, index=df.index)
    output = pd.DataFrame(index=df.index)

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(df.get(column, nan_column), errors="coerce").astype(
            "float64"
        )

        low, high = PLAUSIBLE_RANGES[column]
        implausible = values.notna() & ~values.between(low, high)

        if implausible.any():
            LOGGER.warning(
                "%s: %d implausible value(s) outside [%s, %s] set to missing",
                column,
                int(implausible.sum()),
                low,
                high,
            )

        output[column] = values.mask(implausible)

    for column in BINARY_FEATURES:
        output[column] = _to_binary(df.get(column, nan_column))

    for column in CATEGORICAL_FEATURES + ([SITE_COLUMN] if include_site else []):
        output[column] = _normalise_text(df.get(column, nan_column))

    return output


def audit_data_quality(
    raw: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> dict[str, Any]:
    """Create a machine-readable data-quality report before modelling."""
    mixed_type_columns = []
    for column in raw.columns:
        types = raw[column].dropna().map(type).astype(str).unique()
        if len(types) > 1:
            mixed_type_columns.append(
                {
                    "column": column,
                    "python_types": sorted(types),
                }
            )

    invalid_values = {}
    for column in NUMERIC_FEATURES:
        if column not in raw.columns:
            continue

        values = pd.to_numeric(raw[column], errors="coerce")
        low, high = PLAUSIBLE_RANGES[column]
        invalid = values.notna() & ~values.between(low, high)
        invalid_values[column] = int(invalid.sum())

    duplicate_feature_rows = int(X.duplicated().sum())
    exact_duplicate_rows = int(raw.duplicated().sum())

    report = {
        "n_rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_feature_rows": duplicate_feature_rows,
        "mixed_type_columns": mixed_type_columns,
        "implausible_numeric_values": invalid_values,
        "missing_values": {
            column: int(raw[column].isna().sum()) for column in raw.columns
        },
        "target_counts": {
            str(key): int(value) for key, value in y.value_counts().items()
        },
        "target_positive_rate": round(float(y.mean()), 4),
    }

    return report


def validate_training_frame(X: pd.DataFrame, y: pd.Series) -> None:
    if len(X) < 100:
        raise DataValidationError(
            f"Need at least 100 rows to train reliably, got {len(X)}."
        )

    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 20:
        raise DataValidationError(
            f"Each class needs >= 20 samples, got {counts.to_dict()}."
        )

    empty = [column for column in X.columns if X[column].isna().all()]
    if empty:
        raise DataValidationError(f"Feature(s) with no usable values: {empty}")


def summarise_data(
    X: pd.DataFrame, y: pd.Series, sites: pd.Series | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": int(len(X)),
        "positive_rate": round(float(y.mean()), 4),
        "missing_rate": {
            column: round(float(X[column].isna().mean()), 4) for column in X.columns
        },
        "duplicate_feature_rows": int(X.duplicated().sum()),
    }

    if sites is not None:
        summary["rows_per_site"] = {
            str(key): int(value) for key, value in sites.value_counts().items()
        }
        summary["positive_rate_per_site"] = {
            str(key): round(float(value), 4)
            for key, value in y.groupby(sites).mean().items()
        }

    heavy = [column for column, rate in summary["missing_rate"].items() if rate > 0.3]

    LOGGER.info(
        "Positive rate: %.1f%% | duplicate feature rows: %d",
        100 * summary["positive_rate"],
        summary["duplicate_feature_rows"],
    )

    if heavy:
        LOGGER.info(
            "Features with >30%% missing (kept, with missing-indicators): %s", heavy
        )

    return summary


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def build_preprocessor(include_site: bool) -> ColumnTransformer:
    categorical = CATEGORICAL_FEATURES + ([SITE_COLUMN] if include_site else [])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "binary",
                SimpleImputer(strategy="most_frequent", add_indicator=True),
                BINARY_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def candidate_models(seed: int) -> dict[str, tuple[Any, dict[str, Any]]]:
    models: dict[str, tuple[Any, dict[str, Any]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=seed),
            {
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(max_iter=200, random_state=seed),
            {
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": [2, 3, 4, None],
                "model__max_leaf_nodes": randint(4, 32),
                "model__min_samples_leaf": randint(5, 40),
                "model__l2_regularization": loguniform(1e-3, 10),
            },
        ),
    }

    if module_available("xgboost"):
        from xgboost import XGBClassifier

        models["xgboost"] = (
            XGBClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
                tree_method="hist",
            ),
            {
                "model__n_estimators": randint(100, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(2, 8),
                "model__min_child_weight": randint(1, 10),
                "model__subsample": loguniform(0.6, 1.0),
                "model__colsample_bytree": loguniform(0.6, 1.0),
                "model__reg_lambda": loguniform(1e-3, 10),
            },
        )
        LOGGER.info("XGBoost detected; including it in model comparison.")
    else:
        LOGGER.info("XGBoost not installed; skipping XGBoost model.")

    return models


def tune_models(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: TrainConfig
) -> tuple[str, Pipeline, dict[str, dict[str, Any]]]:
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    summary: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, (estimator, search_space) in candidate_models(cfg.seed).items():
        pipeline = Pipeline(
            [
                ("prep", build_preprocessor(cfg.include_site)),
                ("model", estimator),
            ]
        )

        search = RandomizedSearchCV(
            pipeline,
            search_space,
            n_iter=cfg.n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)
        best_index = search.best_index_

        summary[name] = {
            "cv_roc_auc_mean": float(search.cv_results_["mean_test_score"][best_index]),
            "cv_roc_auc_std": float(search.cv_results_["std_test_score"][best_index]),
            "best_params": {
                key.replace("model__", ""): value
                for key, value in search.best_params_.items()
            },
        }

        fitted[name] = search.best_estimator_

        LOGGER.info(
            "%-24s CV ROC-AUC %.4f +/- %.4f",
            name,
            summary[name]["cv_roc_auc_mean"],
            summary[name]["cv_roc_auc_std"],
        )

    best_name = max(summary, key=lambda name: summary[name]["cv_roc_auc_mean"])
    LOGGER.info("Selected model: %s", best_name)

    return best_name, fitted[best_name], summary


def select_threshold(y_true: np.ndarray, proba: np.ndarray, min_recall: float) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, proba)
    eligible = np.where(true_positive_rate >= min_recall)[0]

    if eligible.size == 0:
        return 0.5

    best = eligible[np.argmin(false_positive_rate[eligible])]
    return float(np.clip(thresholds[best], 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _safe(function: Callable[..., float], *args: Any, **kwargs: Any) -> float | None:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return None


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    prediction = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "roc_auc": _safe(roc_auc_score, y_true, proba),
        "pr_auc": _safe(average_precision_score, y_true, proba),
        "brier": _safe(brier_score_loss, y_true, proba),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: Callable[..., float],
    n_boot: int,
    seed: int,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []

    for _ in range(n_boot):
        indices = rng.integers(0, n, n)

        if len(np.unique(y_true[indices])) < 2:
            continue

        value = metric(y_true[indices], proba[indices])
        if value is not None and not np.isnan(value):
            values.append(float(value))

    if not values:
        return None

    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def subgroup_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    groups: pd.Series,
    min_n: int = 20,
) -> dict[str, Any]:
    keep = ("n", "prevalence", "roc_auc", "recall", "specificity")
    result: dict[str, Any] = {}

    for level in sorted(groups.dropna().unique()):
        mask = (groups == level).to_numpy()

        if mask.sum() < min_n:
            result[str(level)] = {
                "n": int(mask.sum()),
                "note": f"skipped (n < {min_n})",
            }
            continue

        metrics = compute_metrics(y_true[mask], proba[mask], threshold)
        result[str(level)] = {key: metrics[key] for key in keep}

    return result


def cross_site_validation(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sites: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for train_indices, test_indices in LeaveOneGroupOut().split(X, y, groups=sites):
        site = str(sites.iloc[test_indices[0]])
        model = clone(estimator).fit(X.iloc[train_indices], y.iloc[train_indices])
        proba = model.predict_proba(X.iloc[test_indices])[:, 1]
        metrics = compute_metrics(y.iloc[test_indices].to_numpy(), proba, threshold)

        results[site] = {
            key: metrics[key]
            for key in (
                "n",
                "prevalence",
                "roc_auc",
                "pr_auc",
                "recall",
                "specificity",
            )
        }

        LOGGER.info(
            "Leave-site-out %-14s n=%-4d ROC-AUC=%s recall=%.2f",
            site,
            metrics["n"],
            "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}",
            metrics["recall"],
        )

    return results


def save_evaluation_plots(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0, 0])
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0, 0].set_title("ROC curve (test set)")

    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[0, 1])
    axes[0, 1].set_title("Precision-recall curve (test set)")

    fraction_positive, mean_predicted = calibration_curve(
        y_true, proba, n_bins=6, strategy="quantile"
    )

    axes[1, 0].plot(mean_predicted, fraction_positive, "o-", label="model")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    axes[1, 0].set(
        title="Calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed fraction positive",
    )
    axes[1, 0].legend()

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (proba >= threshold).astype(int),
        ax=axes[1, 1],
        display_labels=["No disease", "Disease"],
        colorbar=False,
    )
    axes[1, 1].set_title(f"Confusion matrix @ threshold {threshold:.3f}")

    figure.tight_layout()
    figure.savefig(output_dir / "evaluation.png", dpi=150)
    plt.close(figure)


def save_importance_plot(importance: pd.DataFrame, output_dir: Path) -> None:
    ordered = importance.sort_values("mean_drop_in_auc")
    figure, axis = plt.subplots(figsize=(8, 6))

    axis.barh(
        ordered["feature"],
        ordered["mean_drop_in_auc"],
        xerr=ordered["std"],
        color="#c0392b",
    )
    axis.set(
        title="Permutation importance (test set)",
        xlabel="Drop in ROC-AUC when shuffled",
    )

    figure.tight_layout()
    figure.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(figure)


def save_model_card(
    output_dir: Path,
    model_name: str,
    threshold: float,
    test_metrics: dict[str, Any],
    data_summary: dict[str, Any],
) -> None:
    card = f"""# Heart Disease Risk Prediction Model Card

## Intended use

This model estimates the probability of angiographic heart disease from routine
clinical features. It is intended for educational and portfolio use only and is
**not a medical device**. It must not be used for diagnosis or treatment decisions.

## Target definition

`target = 1` when `num > 0`; otherwise `target = 0`.

## Training data

- Rows: {data_summary["n_rows"]}
- Positive rate: {data_summary["positive_rate"]:.1%}
- Duplicate feature rows detected: {data_summary["duplicate_feature_rows"]}

## Selected model

- Model: `{model_name}`
- Decision threshold: {threshold:.3f}
- Threshold policy: maximize specificity while maintaining out-of-fold recall >= 0.90

## Held-out test performance

| Metric | Value |
|---|---:|
| ROC-AUC | {test_metrics["roc_auc"]:.3f} |
| PR-AUC | {test_metrics["pr_auc"]:.3f} |
| Recall / sensitivity | {test_metrics["recall"]:.3f} |
| Specificity | {test_metrics["specificity"]:.3f} |
| Precision | {test_metrics["precision"]:.3f} |
| Accuracy | {test_metrics["accuracy"]:.3f} |
| Brier score | {test_metrics["brier"]:.3f} |

## Limitations

- Performance varies across clinical source sites.
- The model is trained on historical data and may not generalize to new hospitals,
  devices, populations, or clinical practices.
- Missing values and implausible values are handled statistically, not clinically.
- Feature importance does not establish causality.
- Predicted probabilities require calibration review before clinical interpretation.

## Responsible use

Any real-world use would require clinical review, prospective validation,
bias and fairness assessment, monitoring for data drift, and compliance review.
"""

    (output_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Optional MLflow tracking
# --------------------------------------------------------------------------- #
def log_mlflow_run(
    cfg: TrainConfig,
    model_name: str,
    model_summary: dict[str, dict[str, Any]],
    threshold: float,
    test_metrics: dict[str, Any],
    output_dir: Path,
) -> str | None:
    if not cfg.track_mlflow:
        return None

    if not module_available("mlflow"):
        LOGGER.warning("MLflow is not installed; skipping experiment tracking.")
        return None

    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("heart_disease_prediction")

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(
            {
                "model": model_name,
                "threshold": threshold,
                "test_size": cfg.test_size,
                "cv_folds": cfg.cv_folds,
                "n_iter": cfg.n_iter,
                "min_recall": cfg.min_recall,
                "seed": cfg.seed,
                "n_jobs": cfg.n_jobs,
                "drop_duplicate_rows": cfg.drop_duplicate_rows,
            }
        )

        for key, value in test_metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, value)

        for candidate, result in model_summary.items():
            mlflow.log_metric(f"cv_roc_auc_{candidate}", result["cv_roc_auc_mean"])

        mlflow.log_artifact(str(output_dir / "metrics.json"))
        mlflow.log_artifact(str(output_dir / "metadata.json"))
        mlflow.log_artifact(str(output_dir / "evaluation.png"))
        mlflow.log_artifact(str(output_dir / "feature_importance.png"))
        mlflow.log_artifact(str(output_dir / "MODEL_CARD.md"))
        mlflow.sklearn.log_model(
            sk_model=None,
            artifact_path="model",
            registered_model_name=None,
        )

        LOGGER.info("MLflow run logged: %s", run.info.run_id)
        return run.info.run_id


# --------------------------------------------------------------------------- #
# Training command
# --------------------------------------------------------------------------- #
def run_train(cfg: TrainConfig) -> dict[str, Any]:
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(cfg.data_path)
    y = build_target(raw)

    if cfg.drop_duplicate_rows:
        before = len(raw)
        raw = raw.drop_duplicates(
            subset=ALL_FEATURES + [TARGET_COLUMN, SITE_COLUMN],
            keep="first",
        ).reset_index(drop=True)
        removed = before - len(raw)
        LOGGER.info(
            "Removed %d exact duplicate rows before feature preparation.", removed
        )

    X = prepare_features(raw, include_site=cfg.include_site, strict=True)
    sites = _normalise_text(raw[SITE_COLUMN]) if SITE_COLUMN in raw.columns else None

    validate_training_frame(X, y)
    data_summary = summarise_data(X, y, sites)
    data_quality = audit_data_quality(raw, X, y)

    write_json(output_dir / "data_quality.json", data_quality)

    stratify = y.astype(str) + "_" + sites.astype(str) if sites is not None else y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            stratify=stratify,
            random_state=cfg.seed,
        )
    except ValueError:
        LOGGER.warning(
            "Could not stratify on target x site; stratifying on target only."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            stratify=y,
            random_state=cfg.seed,
        )

    LOGGER.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    best_name, model, model_summary = tune_models(X_train, y_train, cfg)

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    out_of_fold_probability = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=cfg.n_jobs,
    )[:, 1]

    threshold = select_threshold(
        y_train.to_numpy(),
        out_of_fold_probability,
        cfg.min_recall,
    )

    LOGGER.info(
        "Decision threshold %.3f (max specificity with out-of-fold recall >= %.2f)",
        threshold,
        cfg.min_recall,
    )

    y_true = y_test.to_numpy()
    probability = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_true, probability, threshold)

    test_metrics["roc_auc_95ci"] = bootstrap_ci(
        y_true,
        probability,
        roc_auc_score,
        cfg.n_bootstrap,
        cfg.seed,
    )
    test_metrics["pr_auc_95ci"] = bootstrap_ci(
        y_true,
        probability,
        average_precision_score,
        cfg.n_bootstrap,
        cfg.seed,
    )

    baseline = {"majority_class_accuracy": float(max(y_true.mean(), 1 - y_true.mean()))}

    confidence_interval = test_metrics["roc_auc_95ci"]
    LOGGER.info(
        "TEST  ROC-AUC %.3f (95%% CI %s) | recall %.3f | specificity %.3f | accuracy %.3f",
        test_metrics["roc_auc"],
        "n/a"
        if confidence_interval is None
        else f"{confidence_interval[0]:.3f}-{confidence_interval[1]:.3f}",
        test_metrics["recall"],
        test_metrics["specificity"],
        test_metrics["accuracy"],
    )

    subgroups = {"sex": subgroup_metrics(y_true, probability, threshold, X_test["sex"])}

    if sites is not None:
        subgroups["site"] = subgroup_metrics(
            y_true, probability, threshold, sites.loc[X_test.index]
        )

    cross_site = None
    if cfg.cross_site and sites is not None and sites.nunique() > 1:
        cross_site = cross_site_validation(model, X, y, sites, threshold)

    permutation_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=cfg.seed,
        n_jobs=1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_drop_in_auc": permutation_result.importances_mean,
            "std": permutation_result.importances_std,
        }
    ).sort_values("mean_drop_in_auc", ascending=False)

    LOGGER.info(
        "Top features: %s",
        ", ".join(importance["feature"].head(5)),
    )

    save_evaluation_plots(y_true, probability, threshold, output_dir)
    save_importance_plot(importance, output_dir)
    save_model_card(output_dir, best_name, threshold, test_metrics, data_summary)

    categorical_levels = {
        column: sorted(X_train[column].dropna().unique().tolist())
        for column in X.columns
        if column in CATEGORICAL_FEATURES + [SITE_COLUMN]
    }

    bundle = {
        "pipeline": model,
        "threshold": threshold,
        "model_name": best_name,
        "include_site": cfg.include_site,
        "features": list(X.columns),
        "categorical_levels": categorical_levels,
        "sklearn_version": sklearn.__version__,
    }

    joblib.dump(bundle, output_dir / "model.joblib", compress=3)

    report = {
        "selected_model": best_name,
        "threshold": threshold,
        "threshold_policy": (
            f"max specificity subject to out-of-fold recall >= {cfg.min_recall}"
        ),
        "model_selection": model_summary,
        "test": test_metrics,
        "baseline": baseline,
        "subgroups": subgroups,
        "leave_one_site_out": cross_site,
        "feature_importance": importance.to_dict(orient="records"),
    }

    write_json(output_dir / "metrics.json", report)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "target_definition": f"{TARGET_COLUMN} > 0",
        "data": {
            "path": cfg.data_path,
            "sha256": file_sha256(cfg.data_path),
            **data_summary,
        },
        "config": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost_available": module_available("xgboost"),
            "mlflow_available": module_available("mlflow"),
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "binary": BINARY_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }

    write_json(output_dir / "metadata.json", metadata)

    mlflow_run_id = log_mlflow_run(
        cfg,
        best_name,
        model_summary,
        threshold,
        test_metrics,
        output_dir,
    )

    if mlflow_run_id:
        report["mlflow_run_id"] = mlflow_run_id
        write_json(output_dir / "metrics.json", report)

    LOGGER.info("Artifacts written to %s", output_dir.resolve())
    return report


# --------------------------------------------------------------------------- #
# Inference command
# --------------------------------------------------------------------------- #
def read_inference_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame([payload] if isinstance(payload, dict) else payload)

    return pd.read_csv(path, na_values=["?"])


def run_predict(
    model_path: Path,
    input_path: Path,
    output_path: Path | None,
) -> pd.DataFrame:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {model_path} (run `train` first)"
        )

    bundle = joblib.load(model_path)

    if bundle.get("sklearn_version") != sklearn.__version__:
        LOGGER.warning(
            "Model trained with scikit-learn %s, running %s - results may differ.",
            bundle.get("sklearn_version"),
            sklearn.__version__,
        )

    raw = read_inference_input(input_path)

    if raw.empty:
        raise DataValidationError("Input contains no rows.")

    X = prepare_features(
        raw,
        include_site=bundle["include_site"],
        strict=False,
    )[bundle["features"]]

    for column, known_levels in bundle["categorical_levels"].items():
        unseen = set(X[column].dropna().unique()) - set(known_levels)
        if unseen:
            LOGGER.warning(
                "%s: unseen categories %s (encoded as all-zero)",
                column,
                sorted(unseen),
            )

    probability = bundle["pipeline"].predict_proba(X)[:, 1]

    result = pd.DataFrame(
        {
            "probability": np.round(probability, 4),
            "prediction": (probability >= bundle["threshold"]).astype(int),
        }
    )

    if "id" in raw.columns:
        result.insert(0, "id", raw["id"].to_numpy())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        LOGGER.info("Predictions saved to %s", output_path)

    print(
        f"\nthreshold = {bundle['threshold']:.3f}  "
        "(prediction 1 = heart disease likely)"
    )
    print(result.to_string(index=False))
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heart disease prediction pipeline.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="train, evaluate and save a model"
    )
    train_parser.add_argument(
        "--data", type=Path, required=True, help="path to the heart disease CSV"
    )
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--cv-folds", type=int, default=5)
    train_parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="random-search iterations per model",
    )
    train_parser.add_argument(
        "--min-recall",
        type=float,
        default=0.90,
        help="recall target for the decision threshold",
    )
    train_parser.add_argument(
        "--include-site",
        action="store_true",
        help="use the `dataset` column as a feature (not recommended)",
    )
    train_parser.add_argument(
        "--skip-cross-site",
        action="store_true",
        help="skip leave-one-site-out validation",
    )
    train_parser.add_argument("--n-bootstrap", type=int, default=1000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="parallel workers; use 1 on memory-constrained Windows machines",
    )
    train_parser.add_argument(
        "--drop-duplicate-rows",
        action="store_true",
        help="remove exact duplicate rows before training",
    )
    train_parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help="log parameters, metrics, plots and artifacts to MLflow",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="score new patients from a CSV or JSON file"
    )
    predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=".csv, or .json (object or list of objects)",
    )
    predict_parser.add_argument(
        "--model", type=Path, default=Path("artifacts/model.joblib")
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional CSV for predictions",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(None, args.verbose)

    try:
        if args.command == "train":
            if not 0.05 <= args.test_size <= 0.5:
                raise DataValidationError("--test-size must be between 0.05 and 0.5")

            if not 0.5 <= args.min_recall <= 1.0:
                raise DataValidationError("--min-recall must be between 0.5 and 1.0")

            configure_logging(args.output_dir / "run.log", args.verbose)

            run_train(
                TrainConfig(
                    data_path=args.data,
                    output_dir=args.output_dir,
                    test_size=args.test_size,
                    cv_folds=args.cv_folds,
                    n_iter=args.n_iter,
                    min_recall=args.min_recall,
                    include_site=args.include_site,
                    cross_site=not args.skip_cross_site,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    drop_duplicate_rows=args.drop_duplicate_rows,
                    track_mlflow=args.track_mlflow,
                )
            )

        else:
            run_predict(args.model, args.input, args.output)

        return 0

    except (DataValidationError, FileNotFoundError) as error:
        LOGGER.error("%s", error)
        return 2

    except Exception:
        LOGGER.exception("Unexpected failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
