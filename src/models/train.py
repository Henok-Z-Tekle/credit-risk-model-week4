from __future__ import annotations

from typing import Dict, Any

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None


def train_models(X: pd.DataFrame, y: pd.Series, output_dir: str = 'models') -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Logistic Regression grid
    lr = LogisticRegression(max_iter=1000)
    lr_grid = {'C': [0.01, 0.1, 1.0]}
    lr_search = GridSearchCV(lr, lr_grid, cv=3)
    lr_search.fit(X, y)
    lr_best = lr_search.best_estimator_
    y_pred_proba = lr_best.predict_proba(X)[:, 1]
    lr_auc = float(roc_auc_score(y, y_pred_proba))
    results['logistic'] = {'model': lr_best, 'auc': lr_auc}

    # Random Forest grid
    rf = RandomForestClassifier(random_state=0)
    rf_grid = {'n_estimators': [10, 50], 'max_depth': [3, None]}
    rf_search = GridSearchCV(rf, rf_grid, cv=3)
    rf_search.fit(X, y)
    rf_best = rf_search.best_estimator_
    y_pred_proba = rf_best.predict_proba(X)[:, 1]
    rf_auc = float(roc_auc_score(y, y_pred_proba))
    results['random_forest'] = {'model': rf_best, 'auc': rf_auc}

    # Persist best model (highest AUC)
    best_name = max(results.items(), key=lambda kv: kv[1]['auc'])[0]
    best_model = results[best_name]['model']
    model_path = os.path.join(output_dir, 'model_best.joblib')
    joblib.dump(best_model, model_path)
    results['best'] = {'name': best_name, 'path': model_path}

    # Optionally log to MLflow if available
    if mlflow is not None:
        with mlflow.start_run(run_name='train_models'):
            mlflow.log_param('n_samples', int(X.shape[0]))
            mlflow.log_metric('logistic_auc', results['logistic']['auc'])
            mlflow.log_metric('rf_auc', results['random_forest']['auc'])
            mlflow.sklearn.log_model(best_model, 'model')

    return results
