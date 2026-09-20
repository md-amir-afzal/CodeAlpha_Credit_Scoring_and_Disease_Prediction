import pandas as pd, joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

bc=load_breast_cancer()
X=pd.DataFrame(bc.data,columns=bc.feature_names)
y=1-bc.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={
 "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=5000,random_state=42))]),
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
        joblib.dump(model,"disease_prediction_model.joblib")
print("\nBest model saved to disease_prediction_model.joblib")
