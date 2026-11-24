"""Telco customer churn: preprocess, compare models, tune, and save the final model.

Run from anywhere:
    python customer_churn.py

Reads  : Telco-Customer-Churn.csv                      (next to this file)
Writes : churn_model.pkl, scaler.pkl, threshold.pkl,    (next to this file)
         model_metrics.csv

Exploratory analysis (histograms, churn-rate tables, correlation and log-odds
checks) stays in customer_churn.ipynb. This script is the reproducible
training pipeline.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is only needed for the baseline comparison table
    XGBClassifier = None

log = logging.getLogger("customer_churn")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "threshold.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Decision cutoff: customers with churn probability >= this are flagged.
# Lower than 0.5 on purpose: missing a churner costs more than a false alarm.
FINAL_THRESHOLD = 0.40
THRESHOLD_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

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


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and encode the raw Telco frame. Returns (features, target)."""
    df = df.copy()

    # customerID is only an identifier. TotalCharges is roughly
    # tenure x MonthlyCharges (redundant), and has blank-string rows, so drop it.
    df = df.drop(columns=["customerID", "TotalCharges"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # "No internet service" / "No phone service" carry the same meaning as "No".
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)

    # Multi-category columns have no natural order -> one-hot.
    # drop_first avoids the dummy-variable trap.
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True, dtype=int)

    X = df.drop(columns="Churn")
    y = df["Churn"]

    if X.isna().any().any() or y.isna().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Unexpected missing values after preprocessing: {bad}")
    return X, y


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def make_pipeline(clf) -> Pipeline:
    """Scaler + classifier, so scaling is fit on training data only (also inside CV)."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_baselines(y_train: pd.Series) -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        # scale_pos_weight is XGBoost's version of class_weight (stay : churn ratio)
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        log.warning("xgboost is not installed; skipping XGBoost in the comparison.")
    return {name: make_pipeline(clf) for name, clf in models.items()}


def evaluate(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "recall": round(recall_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred), 3),
        "f1": round(f1_score(y_true, pred), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
    }


def compare_models(models, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "threshold": 0.5, **evaluate(y_test, proba)})
    return (
        pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)
    )


def tune_logistic_regression(X_train, y_train) -> Pipeline:
    """Grid-search the regularisation strength C by cross-validated ROC-AUC."""
    search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
        param_grid={"clf__C": [0.05, 0.1, 0.5, 1, 2]},
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log.info(
        "Best params: %s | CV ROC-AUC: %.3f", search.best_params_, search.best_score_
    )
    return search.best_estimator_


def threshold_table(y_true, proba, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "recall": round(recall_score(y_true, pred), 3),
                "precision": round(precision_score(y_true, pred), 3),
                "caught": int(((pred == 1) & (y_true == 1)).sum()),
                "missed": int(((pred == 0) & (y_true == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def odds_ratios(best_pipe: Pipeline, feature_names) -> pd.Series:
    """Odds ratio per 1 standard deviation of each feature (>1 raises churn odds)."""
    coefs = best_pipe.named_steps["clf"].coef_[0]
    return (
        pd.Series(np.exp(coefs), index=feature_names)
        .sort_values(ascending=False)
        .round(2)
    )


# --------------------------------------------------------------------------- #
# Saving and inference
# --------------------------------------------------------------------------- #
def save_artifacts(best_pipe: Pipeline, threshold: float) -> None:
    """Save model, scaler and threshold as separate files (same names as before)."""
    joblib.dump(best_pipe.named_steps["clf"], MODEL_PATH)
    joblib.dump(best_pipe.named_steps["scaler"], SCALER_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)


def predict_churn(X_new: pd.DataFrame) -> pd.DataFrame:
    """Score customers with the saved artifacts.

    Pass RAW (unscaled) features in the encoded format used for training;
    this function scales them. Columns are aligned to the training columns:
    extras are dropped and missing one-hot columns become 0. (For one raw
    customer, call pd.get_dummies WITHOUT drop_first and this alignment
    removes the baseline levels for you.)
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    X_aligned = X_new.reindex(columns=scaler.feature_names_in_, fill_value=0)
    proba = model.predict_proba(scaler.transform(X_aligned))[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": proba.round(3),
            "churn_prediction": (proba >= threshold).astype(int),
        },
        index=X_new.index,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_data()
    X, y = preprocess(df)
    log.info(
        "Rows: %d | features: %d | churn rate: %.1f%%",
        len(X),
        X.shape[1],
        y.mean() * 100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    log.info("\n=== Baseline models (threshold 0.5) ===")
    baseline_metrics = compare_models(
        build_baselines(y_train), X_train, y_train, X_test, y_test
    )
    log.info("\n%s", baseline_metrics.to_string(index=False))

    log.info("\n=== Tuning Logistic Regression ===")
    best_pipe = tune_logistic_regression(X_train, y_train)

    # Choose the cutoff from out-of-fold predictions on TRAIN data, so the
    # test set is not used to pick the threshold.
    oof_proba = cross_val_predict(
        best_pipe, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]
    log.info("\n=== Threshold sweep (out-of-fold, train data) ===")
    log.info("\n%s", threshold_table(y_train, oof_proba).to_string(index=False))

    log.info("\n=== Final model on the test set (threshold %.2f) ===", FINAL_THRESHOLD)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    final_metrics = evaluate(y_test, test_proba, FINAL_THRESHOLD)
    log.info("%s", final_metrics)

    final_row = pd.DataFrame(
        [
            {
                "model": "Logistic Regression (tuned, final)",
                "threshold": FINAL_THRESHOLD,
                **final_metrics,
            }
        ]
    )
    pd.concat([baseline_metrics, final_row], ignore_index=True).to_csv(
        METRICS_PATH, index=False
    )

    save_artifacts(best_pipe, FINAL_THRESHOLD)
    log.info(
        "Saved: %s, %s, %s, %s",
        MODEL_PATH.name,
        SCALER_PATH.name,
        THRESHOLD_PATH.name,
        METRICS_PATH.name,
    )

    # Sanity check: reloaded artifacts must reproduce the in-memory model.
    reloaded = predict_churn(X_test)["churn_probability"].to_numpy()
    np.testing.assert_allclose(reloaded, test_proba, atol=1e-3)
    sample = predict_churn(X_test.head(5))
    sample["actual"] = y_test.head(5).to_numpy()
    log.info("\n=== Reload check (first 5 test customers) ===\n%s", sample.to_string())

    log.info(
        "\n=== Odds ratios (per 1 SD) ===\n%s",
        odds_ratios(best_pipe, X.columns).to_string(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
