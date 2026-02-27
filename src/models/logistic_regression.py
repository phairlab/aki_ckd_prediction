"""
Logistic regression training and prediction.
"""

from sklearn.linear_model import LogisticRegression


def train_logreg(X_train, y_train, random_state=1202):
    """Fit a LogisticRegression and return it."""
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def predict_logreg(model, X_test):
    """Return (y_pred, probas_positive_class)."""
    y_pred = model.predict(X_test).astype("int8")
    probas = model.predict_proba(X_test)[:, 1]
    return y_pred, probas
