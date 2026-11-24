# machine-learning-portfolio

End-to-end machine learning projects covering classification, sequence modeling, and deployable applications.

## Projects

| Project | Description |
|---|---|
| [heart_disease_prediction](heart_disease_prediction) | Leakage-safe heart-disease risk prediction pipeline using XGBoost, scikit-learn, calibration, permutation importance, leave-one-site-out validation, and a FastAPI inference service |
| [emotion_Intelligence](emotion_Intelligence) | Six-class emotion classifier using a BiGRU model with a Streamlit interface |
| [customer-churn-prediction](customer-churn-prediction) | Customer churn prediction model with evaluation and Streamlit UI |
| [Food-prediction](Food-prediction) | Food prediction model |

## Heart Disease Prediction

- Dataset: 100,000 multi-site clinical records
- Selected model: XGBoost
- Test ROC-AUC: 0.970
- Test recall: 89.7%
- Test specificity: 92.2%
- Evaluation: calibration, PR-AUC, bootstrap confidence intervals, subgroup metrics, permutation importance, and leave-one-site-out validation

## Run the Heart Disease API

```bash
cd heart_disease_prediction
pip install -r requirements.txt
python backend.py
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Run an app

```bash
python -m pip install -r requirements.txt
python -m streamlit run heart_disease_prediction/app.py
```
