"""
XGBoost training, prediction, and feature importance helpers.
"""

import numpy as np
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, random_state=1202):
    """Fit an XGBClassifier and return it."""
    clf = XGBClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def predict_xgboost(model, X_test):
    """Return (y_pred, probas_positive_class)."""
    y_pred = model.predict(X_test).astype("int8")
    probas = model.predict_proba(X_test)[:, 1]
    return y_pred, probas


def get_feature_importances(model, feature_names):
    """Return list of (feature_name, importance) sorted descending."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in indices]
