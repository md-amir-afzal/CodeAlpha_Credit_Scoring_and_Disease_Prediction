import joblib
from sklearn.datasets import load_breast_cancer

try:
    model=joblib.load("disease_prediction_model.joblib")
except FileNotFoundError:
    raise SystemExit("Model not found. Run: python train_model.py")

bc=load_breast_cancer()
sample=bc.data[[0]]
pred=model.predict(sample)[0]
prob=model.predict_proba(sample)[0,1]
print("Model output:", "Disease class predicted" if pred==1 else "No disease class predicted")
print(f"Predicted probability for disease class: {prob:.2%}")
print("Educational use only; this is not a medical diagnosis.")
