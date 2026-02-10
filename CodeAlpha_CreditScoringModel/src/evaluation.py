from sklearn.metrics import roc_auc_score
from models import X_train, X_test, y_train, y_test, logreg, rf, gb, scaler

train_preds = (
    0.4*logreg.predict_proba(scaler.transform(X_train))[:,1] +
    0.3*rf.predict_proba(X_train)[:,1] +
    0.3*gb.predict_proba(X_train)[:,1]
)

test_preds = (
    0.4*logreg.predict_proba(scaler.transform(X_test))[:,1] +
    0.3*rf.predict_proba(X_test)[:,1] +
    0.3*gb.predict_proba(X_test)[:,1]
)

print("Train ROC-AUC:", roc_auc_score(y_train, train_preds))
print("Test ROC-AUC :", roc_auc_score(y_test, test_preds))
