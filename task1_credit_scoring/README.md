# CodeAlpha Task 1 — Credit Scoring Model

## Objective
Predict an individual's creditworthiness from financial-history features using classification models.

## Models
- Logistic Regression with feature scaling
- Random Forest with class balancing

## Evaluation
Accuracy, Precision, Recall, F1-Score and ROC-AUC are reported.

## Dataset
`credit_data.csv` is a reproducible synthetic dataset created for this internship project because the supplied task sheet does not mandate one specific credit dataset. It contains income, debt, payment history and related financial features.

## Run
```bash
pip install -r ../requirements.txt
python train_model.py
python predict.py
```
