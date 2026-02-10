import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from models import X_train, X_test, y_train, y_test, logreg, rf, gb, scaler

# BLENDING
p_log = logreg.predict_proba(scaler.transform(X_test))[:,1]
p_rf = rf.predict_proba(X_test)[:,1]
p_gb = gb.predict_proba(X_test)[:,1]

p_blend = 0.4*p_log + 0.3*p_rf + 0.3*p_gb
print("Blended ROC-AUC:", roc_auc_score(y_test, p_blend))

# STACKING
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((X_train.shape[0], 3))
meta_test = np.zeros((X_test.shape[0], 3))

for i, model in enumerate([logreg, rf, gb]):
    for tr, val in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr], X_train.iloc[val]
        y_tr = y_train.iloc[tr]

        if i == 0:
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        model.fit(X_tr, y_tr)
        meta_train[val, i] = model.predict_proba(X_val)[:,1]

    meta_test[:,i] = (
        model.predict_proba(scaler.transform(X_test))[:,1]
        if i == 0 else model.predict_proba(X_test)[:,1]
    )

meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_train, y_train)
stacked_preds = meta_model.predict_proba(meta_test)[:,1]

print("Stacked ROC-AUC:", roc_auc_score(y_test, stacked_preds))
