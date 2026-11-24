# Heart Disease Risk Prediction Model Card

## Intended use

This model estimates the probability of angiographic heart disease from routine
clinical features. It is intended for educational and portfolio use only and is
**not a medical device**. It must not be used for diagnosis or treatment decisions.

## Target definition

`target = 1` when `num > 0`; otherwise `target = 0`.

## Training data

- Rows: 100000
- Positive rate: 55.4%
- Duplicate feature rows detected: 2052

## Selected model

- Model: `xgboost`
- Decision threshold: 0.652
- Threshold policy: maximize specificity while maintaining out-of-fold recall >= 0.90

## Held-out test performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.970 |
| PR-AUC | 0.973 |
| Recall / sensitivity | 0.897 |
| Specificity | 0.922 |
| Precision | 0.934 |
| Accuracy | 0.908 |
| Brier score | 0.064 |

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
