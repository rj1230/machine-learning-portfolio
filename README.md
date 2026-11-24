<div align="center">

# Machine Learning Portfolio

### End-to-end machine-learning systems for prediction, sequence modeling, and deployment

A curated collection of machine-learning projects covering clinical risk prediction, natural-language classification, customer analytics, and model deployment.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-EC6C2C)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference_Service-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Apps-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## Overview

This repository demonstrates the complete machine-learning development lifecycle: problem framing, data preparation, model development, rigorous evaluation, interpretation, and deployment.

The projects are designed to move beyond standalone notebooks. They emphasize leakage-safe validation, calibrated probabilistic outputs, robustness analysis, feature attribution, and production-oriented serving through APIs or interactive interfaces.

> **Objective:** Build machine-learning systems that are measurable, interpretable, reproducible, and ready for practical use.

## Projects

| Project | Domain | Description |
|---|---|---|
| [Heart Disease Prediction](heart_disease_prediction) | Clinical risk prediction | Leakage-safe heart-disease risk-prediction pipeline using XGBoost, calibration analysis, subgroup evaluation, permutation importance, leave-one-site-out validation, and a FastAPI inference service |
| [Emotion Intelligence](emotion_Intelligence) | Natural language processing | Six-class emotion-classification system using a BiGRU sequence model and a Streamlit interface |
| [Customer Churn Prediction](customer-churn-prediction) | Customer analytics | Churn-prediction model with model evaluation and an interactive Streamlit user interface |
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

The evaluation strategy is designed to assess more than a single aggregate accuracy score:

- **Calibration analysis** — evaluates whether predicted probabilities align with observed outcomes.
- **PR-AUC** — measures performance under class imbalance and emphasizes precision-recall trade-offs.
- **Bootstrap confidence intervals** — quantifies uncertainty around reported performance metrics.
- **Subgroup analysis** — examines model behavior across clinically relevant patient groups.
- **Permutation importance** — identifies the relative contribution of input features to model predictions.
- **Leave-one-site-out validation** — tests generalization to unseen clinical sites and reduces the risk of site-specific overfitting.

This approach supports a more trustworthy assessment of model performance than evaluation on a single randomly split test set.

## API Deployment

The heart-disease project includes a FastAPI service for serving model predictions through a documented HTTP interface.

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
cd heart_disease_prediction
python -m pip install -r requirements.txt
```

### Start the API

```bash
python backend.py
```

### Access the documentation

Once the server is running, open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Streamlit Application

The heart-disease project also includes a Streamlit-based interface for interactive experimentation and demonstration.

```bash
python -m pip install -r requirements.txt
python -m streamlit run heart_disease_prediction/app.py
```

## Technology Stack

| Layer | Technologies |
|---|---|
| Machine learning | scikit-learn, XGBoost |
| Deep learning | BiGRU |
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
├── emotion_Intelligence/
├── customer-churn-prediction/
├── Food-prediction/
└── README.md
```

## Engineering Principles

- **End-to-end development** — Projects span data preparation, modeling, evaluation, interpretation, and deployment.
- **Leakage-safe validation** — Validation workflows are designed to prevent information leakage and inflated performance estimates.
- **Evaluation beyond accuracy** — Models are assessed using calibration, uncertainty, subgroup behavior, and feature attribution.
- **Deployment readiness** — Selected projects expose predictions through APIs or interactive applications.
- **Reproducibility** — Dependencies, commands, and execution steps are documented for local reproduction.

## License

This repository is intended for educational and portfolio purposes. Add a license file before public distribution if reuse permissions need to be defined.

## Author

**Raj Rajput**  
GitHub: [@rj1230](https://github.com/rj1230)
