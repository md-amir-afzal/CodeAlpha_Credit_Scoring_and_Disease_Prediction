import joblib, pandas as pd

model=joblib.load("credit_scoring_model.joblib")
sample=pd.DataFrame([{
    "income":65000,"debt":25000,"credit_history_months":72,"late_payments":1,
    "loan_count":2,"employment_years":4,"savings":120000,
    "credit_utilization":28,"monthly_expenses":22000,"existing_credit":90000
}])
pred=model.predict(sample)[0]
prob=model.predict_proba(sample)[0,1]
print("Prediction:", "Creditworthy" if pred==1 else "Higher Risk")
print(f"Probability of creditworthy class: {prob:.2%}")
