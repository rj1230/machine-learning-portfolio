# machine-learning-portfolio

End-to-end machine learning projects: customer churn prediction, heart disease risk, BiGRU emotion classification, and food prediction. Reproducible training, evaluation, and deployable apps.

## Projects

| Project | Description |
|---|---|
| [heart_disease_prediction](heart_disease_prediction) | Leakage-safe heart disease risk pipeline (XGBoost, scikit-learn) with a Streamlit dashboard |
| [emotion_Intelligence](emotion_Intelligence) | Six-class emotion classifier (BiGRU) with a Streamlit app |
| [customer-churn-prediction](customer-churn-prediction) | Customer churn prediction |
| [Food-prediction](Food-prediction) | Food prediction model |

## Run an app

```bash
python -m pip install -r requirements.txt
python -m streamlit run heart_disease_prediction/app.py
```
