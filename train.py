"""
Train + evaluate.

Classifiers:
  - 'logreg' : sklearn LogisticRegression (used for TF-IDF baseline)
  - 'mlp'    : 2-layer MLP with dropout (used for partial ablations)
  - 'xgboost': XGBoost GBT (used for the full hybrid model)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.neural_network import MLPClassifier

from .utils import log


def _make_classifier(name, seed):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256, 64), activation="relu",
                             alpha=1e-4, learning_rate_init=1e-3, max_iter=80,
                             early_stopping=True, random_state=seed)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 eval_metric="logloss", n_jobs=-1, random_state=seed)
        except ImportError:
            log("xgboost not installed — falling back to MLP.")
            return _make_classifier("mlp", seed)
    raise ValueError(name)


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       classifier="mlp", seed=42):
    clf = _make_classifier(classifier, seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else 0.0,
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "classifier": classifier,
    }
    return metrics, clf
