# Machine Learning Portfolio

End-to-end machine learning projects demonstrating model development, rigorous evaluation, interpretability, reproducibility, and deployable applications.

The portfolio includes supervised classification, deep learning, risk prediction, and interactive application development using Python, scikit-learn, XGBoost, TensorFlow/Keras, Streamlit, and FastAPI.

## Featured Project

### Heart Disease Risk Prediction

A leakage-safe, end-to-end machine-learning pipeline that predicts the likelihood of angiographic heart disease from clinical features. The project includes data-quality auditing, model comparison, threshold optimization, calibration, interpretability, cross-site validation, and a FastAPI inference service.

**Key results**

| Metric | Held-out test performance |
|---|---:|
| ROC-AUC | 0.970 |
| PR-AUC | 0.973 |
| Recall / sensitivity | 89.7% |
| Specificity | 92.2% |
| Accuracy | 90.8% |
| F1 score | 91.6% |
| Decision threshold | 0.652 |

**Highlights**

- Processed 100,000 multi-site clinical records with data validation, missing-value handling, and physiologically implausible-value detection.
- Compared logistic regression, random forest, histogram gradient boosting, and XGBoost using stratified cross-validation and randomized hyperparameter search.
- Selected XGBoost, which achieved a cross-validated ROC-AUC of 0.971.
- Optimized the classification threshold on out-of-fold predictions to maintain at least 90% recall while maximizing specificity.
- Evaluated model reliability using calibration curves, Brier score, PR-AUC, bootstrap confidence intervals, and confusion-matrix analysis.
- Assessed generalization with leave-one-site-out validation across four clinical cohorts.
- Generated permutation feature importance and subgroup performance metrics by sex and clinical site.
- Produced reproducible artifacts, including a serialized model bundle, metrics report, data-quality report, evaluation plots, and model card.
- Built a FastAPI inference service with input validation, single and batch prediction endpoints, health checks, and interactive API documentation.

**Top predictive features**

`cp`, `exang`, `thalch`, `oldpeak`, `slope`, and `ca`

These features are predictive associations identified through permutation importance and do not establish causality.

## Project Portfolio

| Project | Description | Key technologies |
|---|---|---|
| [Heart Disease Prediction](heart_disease_prediction) | End-to-end clinical risk-prediction pipeline with robust evaluation, interpretability, and a FastAPI inference service | Python, scikit-learn, XGBoost, Pandas, FastAPI, Joblib |
| [Emotion Intelligence](emotion_Intelligence) | Six-class emotion classification system using a Bidirectional GRU neural network and Streamlit interface | Python, TensorFlow/Keras, BiGRU, Streamlit |
| [Customer Churn Prediction](customer-churn-prediction) | Customer churn classification model with evaluation metrics and an interactive Streamlit dashboard | Python, scikit-learn, Pandas, Streamlit |
| [Food Prediction](Food-prediction) | Food prediction model | Python, scikit-learn |

## Repository Structure

```text
machine-learning-portfolio/
├── heart_disease_prediction/
│   ├── main.py                 # Training, evaluation, and artifact generation
│   ├── backend.py              # FastAPI inference service
│   ├── app.py                  # Streamlit dashboard
│   ├── artifacts/              # Model, metrics, plots, and reports
│   └── heart_disease.csv       # Dataset
├── emotion_Intelligence/
│   ├── main.py                 # BiGRU training pipeline
│   ├── app.py                  # Streamlit application
│   └── artifacts/              # Trained model and tokenizer
├── customer-churn-prediction/
│   ├── customer_churn.py       # Churn modeling pipeline
│   ├── ui.py                   # Streamlit dashboard
│   └── model_metrics.csv       # Evaluation results
└── Food-prediction/
    └── ...                     # Food prediction project files
```

## Technical Skills

- **Machine Learning:** Classification, model selection, hyperparameter tuning, cross-validation, threshold optimization, calibration, and feature importance
- **Deep Learning:** Bidirectional GRU, sequence modeling, text classification, and Keras model development
- **Data Engineering:** Data validation, cleaning, missing-value handling, feature engineering, and reproducible preprocessing
- **Evaluation:** ROC-AUC, PR-AUC, precision, recall, specificity, F1 score, Brier score, confusion matrices, bootstrap confidence intervals, and subgroup analysis
- **Interpretability:** Permutation importance, calibration analysis, and model cards
- **Deployment:** FastAPI, REST APIs, Pydantic validation, Streamlit, and Joblib model serialization
- **Tools:** Python, Pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Git, and GitHub

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/rj1230/machine-learning-portfolio.git
cd machine-learning-portfolio
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Heart Disease API

The heart-disease project includes a FastAPI service that loads the trained model and returns a risk probability and classification for patient input.

```bash
cd heart_disease_prediction
python backend.py
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides Swagger UI-based interactive documentation at `/docs`, allowing endpoints to be explored and tested from the browser. [75][79]

### Example prediction request

```json
{
  "age": 58,
  "sex": "male",
  "cp": "asymptomatic",
  "trestbps": 140,
  "chol": 240,
  "fbs": false,
  "restecg": "normal",
  "thalch": 150,
  "exang": true,
  "oldpeak": 2.3,
  "slope": "flat",
  "ca": 2,
  "thal": "reversible_defect"
}
```

### Example response

```json
{
  "probability": 0.8421,
  "prediction": 1,
  "risk_label": "high_risk",
  "threshold": 0.652,
  "model_name": "xgboost"
}
```

## Run the Streamlit Applications

### Heart disease dashboard

```bash
cd heart_disease_prediction
python -m streamlit run app.py
```

### Emotion intelligence app

```bash
cd emotion_Intelligence
python -m streamlit run app.py
```

### Customer churn dashboard

```bash
cd customer-churn-prediction
python -m streamlit run ui.py
```

## Retrain the Heart Disease Model

```bash
cd heart_disease_prediction
python main.py train --data heart_disease.csv --n-iter 10 --n-jobs 1
```

Optional: remove exact duplicate rows before training.

```bash
python main.py train --data heart_disease.csv --n-iter 10 --n-jobs 1 --drop-duplicate-rows
```

## Responsible AI

The heart-disease model is intended for educational and portfolio purposes only. It is **not a medical device** and must not be used for diagnosis, treatment, or clinical decision-making.

Real-world deployment would require clinical validation, bias and fairness assessment, monitoring for data drift, regulatory review, and ongoing model governance.
