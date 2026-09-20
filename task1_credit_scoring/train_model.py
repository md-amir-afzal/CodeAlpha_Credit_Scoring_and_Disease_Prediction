import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

FEATURES=["income","debt","credit_history_months","late_payments","loan_count",
          "employment_years","savings","credit_utilization","monthly_expenses","existing_credit"]

# Reproducible educational credit dataset.
X_raw,y=make_classification(n_samples=2500,n_features=10,n_informative=7,n_redundant=2,
                            weights=[0.58,0.42],class_sep=1.15,random_state=42)
df=pd.DataFrame(X_raw,columns=FEATURES)
df["creditworthy"]=y
df.to_csv("credit_data.csv",index=False)

X=df[FEATURES]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={
 "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,random_state=42))]),
 "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight="balanced")
}
best_auc=-1
for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
    auc=roc_auc_score(y_test,proba)
    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y_test,pred):.4f}")
    print(f"Precision: {precision_score(y_test,pred):.4f}")
    print(f"Recall   : {recall_score(y_test,pred):.4f}")
    print(f"F1       : {f1_score(y_test,pred):.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    if auc>best_auc:
        best_auc=auc
        joblib.dump(model,"credit_scoring_model.joblib")
print("\nBest model saved to credit_scoring_model.joblib")
