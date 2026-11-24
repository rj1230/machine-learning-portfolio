<div align="center">

# Machine Learning Portfolio

### End-to-end machine-learning systems for prediction, sequence modeling, and deployment

A curated collection of machine-learning projects covering clinical risk prediction, delivery-time forecasting, customer analytics, natural-language classification, and model deployment.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-EC6C2C)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference_Service-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demos-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## Overview

This repository demonstrates the complete machine-learning development lifecycle: problem framing, data preparation, model development, evaluation, interpretation, and deployment.

The projects emphasize leakage-safe validation, calibrated probabilistic outputs, robustness analysis, feature attribution, and production-oriented serving through APIs or interactive web applications.

> **Objective:** Build machine-learning systems that are measurable, interpretable, reproducible, and ready for practical use.

## Live Demos

| Project | Live demo |
|---|---|
| Heart Disease Prediction | [Open Streamlit app](https://machine-learning-portfolio-yvplwjtixokwsaowpeaawi.streamlit.app/) |
| Delivery Time Predictor | [Open Streamlit app](https://machine-learning-portfolio-yvplwjtixokwsaowpeaawi.streamlit.app/) |
| Customer Churn Prediction | [Open Streamlit app](https://machine-learning-portfolio-yvplwjtixokwsaowpeaawi.streamlit.app/) |
| Emotion Intelligence | [Open Streamlit app](https://machine-learning-portfolio-ldcbeszryex4ncjmvpdesm.streamlit.app/) |

> The Streamlit applications provide interactive interfaces for exploring model inputs, predictions, and project workflows.

## Projects

| Project | Domain | Description |
|---|---|---|
| [Heart Disease Prediction](heart_disease_prediction) | Clinical risk prediction | Leakage-safe heart-disease risk-prediction pipeline using XGBoost, calibration analysis, subgroup evaluation, permutation importance, leave-one-site-out validation, and a FastAPI inference service |
| [Delivery Time Predictor](delivery-time-predictor) | Regression / operations | Delivery-time prediction model with an interactive Streamlit interface |
| [Customer Churn Prediction](customer-churn-prediction) | Customer analytics | Churn-prediction model with model evaluation and an interactive Streamlit user interface |
| [Emotion Intelligence](emotion_Intelligence) | Natural language processing | Six-class emotion-classification system using a BiGRU sequence model and a Streamlit interface |
| [Food Prediction](Food-prediction) | Classification | Food-category prediction model |

## Featured Project: Heart Disease Prediction

### Problem statement

Cardiovascular disease risk prediction requires models that are not only accurate but also reliable across patient populations and clinical sites. This project develops a leakage-safe prediction pipeline for heart-disease risk using multi-site clinical records.

### Dataset

| Attribute | Details |
|---|---|
| Dataset size | 100,000 clinical records |
| Data structure | Multi-site clinical dataset |
| Validation strategy | Leave-one-site-out validation |
| Primary task | Binary heart-disease risk prediction |

### Modeling approach

The project uses an XGBoost-based classification pipeline with structured clinical features. Model development follows a leakage-safe workflow, ensuring that preprocessing and feature-selection decisions do not leak information from validation or test data into training.

### Performance

| Metric | Result |
|---|---:|
| Selected model | XGBoost |
| Test ROC-AUC | **0.970** |
| Test recall | **89.7%** |
| Test specificity | **92.2%** |

### Evaluation methodology

The evaluation strategy assesses more than aggregate accuracy:

- **Calibration analysis** — evaluates whether predicted probabilities align with observed outcomes.
- **PR-AUC** — measures performance under class imbalance and highlights precision-recall trade-offs.
- **Bootstrap confidence intervals** — quantifies uncertainty around reported performance metrics.
- **Subgroup analysis** — examines model behavior across clinically relevant patient groups.
- **Permutation importance** — identifies the relative contribution of input features to predictions.
- **Leave-one-site-out validation** — tests generalization to unseen clinical sites and reduces the risk of site-specific overfitting.

## Heart Disease API

The heart-disease project includes a FastAPI service for serving model predictions through a documented HTTP interface.

### Installation

```bash
cd heart_disease_prediction
python -m pip install -r requirements.txt
```

### Start the API

```bash
python backend.py
```

### API documentation

```text
http://127.0.0.1:8000/docs
```

## Run Applications Locally

### Heart Disease Prediction

```bash
cd heart_disease_prediction
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Delivery Time Predictor

```bash
cd delivery-time-predictor
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Customer Churn Prediction

```bash
cd customer-churn-prediction
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Emotion Intelligence

```bash
cd emotion_Intelligence
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Technology Stack

| Layer | Technologies |
|---|---|
| Machine learning | scikit-learn, XGBoost |
| Deep learning | BiGRU |
| Regression | Delivery-time prediction models |
| API serving | FastAPI |
| User interfaces | Streamlit |
| Model evaluation | Calibration, PR-AUC, bootstrap confidence intervals, permutation importance |
| Validation | Leave-one-site-out validation |

## Repository Structure

```text
machine-learning-portfolio/
├── heart_disease_prediction/
│   ├── backend.py
│   ├── app.py
│   └── requirements.txt
├── delivery-time-predictor/
├── customer-churn-prediction/
├── emotion_Intelligence/
├── Food-prediction/
└── README.md
```

## Engineering Principles

- **End-to-end development** — Projects span data preparation, modeling, evaluation, interpretation, and deployment.
- **Leakage-safe validation** — Validation workflows are designed to prevent information leakage and inflated performance estimates.
- **Evaluation beyond accuracy** — Models are assessed using calibration, uncertainty, subgroup behavior, and feature attribution.
- **Interactive deployment** — Models are made accessible through Streamlit applications and, where applicable, FastAPI services.
- **Reproducibility** — Dependencies, commands, and execution steps are documented for local reproduction.

## License

This repository is intended for educational and portfolio purposes. Add a license file before public distribution if reuse permissions need to be defined.

## Author

**Raj Rajput**  
GitHub: [@rj1230](https://github.com/rj1230)
