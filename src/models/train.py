from __future__ import annotations

from typing import Dict, Any, Optional

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
except Exception:
    mlflow = None


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = 'models',
    test_size: float = 0.2,
    random_state: int = 42,
    mlflow_experiment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train candidate models with a proper train/test split, evaluate and optionally log to MLflow.

    IMPORTANT: Before calling this function, ensure that your proxy target (e.g., 'is_high_risk')
    is explicitly merged into your feature DataFrame and passed as the target `y`.

    Returns a dict with trained models, test metrics and persisted model path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Split data to ensure honest evaluation
    stratify = y if (hasattr(y, 'nunique') and y.nunique() == 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    results: Dict[str, Any] = {}

    # Logistic Regression grid (train on train split)
    lr = LogisticRegression(max_iter=1000)
    lr_grid = {'C': [0.01, 0.1, 1.0]}
    lr_search = GridSearchCV(lr, lr_grid, cv=3)
    lr_search.fit(X_train, y_train)
    lr_best = lr_search.best_estimator_
    y_proba_lr = lr_best.predict_proba(X_test)[:, 1]
    y_pred_lr = lr_best.predict(X_test)
    results['logistic'] = {
        'model': lr_best,
        'auc': float(roc_auc_score(y_test, y_proba_lr)),
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_lr, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_lr, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist(),
    }

    # Random Forest grid
    rf = RandomForestClassifier(random_state=random_state)
    rf_grid = {'n_estimators': [10, 50], 'max_depth': [3, None]}
    rf_search = GridSearchCV(rf, rf_grid, cv=3)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    y_proba_rf = rf_best.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_best.predict(X_test)
    results['random_forest'] = {
        'model': rf_best,
        'auc': float(roc_auc_score(y_test, y_proba_rf)),
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_rf, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_rf, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
    }

    # Select best model by AUC on test set
    best_name = max(results.items(), key=lambda kv: kv[1]['auc'])[0]
    best_model = results[best_name]['model']

    # Retrain selected best model on the full dataset for deployment
    best_model.fit(X, y)
    model_path = os.path.join(output_dir, 'model_best.joblib')
    joblib.dump(best_model, model_path)
    results['best'] = {'name': best_name, 'path': model_path}

    # Optionally log to MLflow if available
    if mlflow is not None:
        mlflow.set_experiment(mlflow_experiment or os.environ.get('MLFLOW_EXPERIMENT', 'credit-risk'))
        with mlflow.start_run(run_name='train_models') as run:
            mlflow.log_param('n_samples', int(X.shape[0]))
            mlflow.log_param('test_size', float(test_size))
            mlflow.log_param('random_state', int(random_state))
            # Log metrics for each candidate on test set
            for name, info in results.items():
                if name in ['logistic', 'random_forest']:
                    mlflow.log_metric(f'{name}_auc', float(info['auc']))
                    mlflow.log_metric(f'{name}_accuracy', float(info['accuracy']))
                    mlflow.log_metric(f'{name}_precision', float(info['precision']))
                    mlflow.log_metric(f'{name}_recall', float(info['recall']))
                    mlflow.log_metric(f'{name}_f1', float(info['f1']))

            # Log best model with signature if possible
            try:
                signature = infer_signature(X_test, best_model.predict_proba(X_test)[:, 1])
            except Exception:
                signature = None

            try:
                register_name = os.environ.get('MLFLOW_REGISTER_MODEL')
                if register_name:
                    mlflow.sklearn.log_model(best_model, 'model', registered_model_name=register_name)
                else:
                    mlflow.sklearn.log_model(best_model, 'model')
            except Exception:
                # fall back to simple logging
                try:
                    mlflow.sklearn.log_model(best_model, 'model')
                except Exception:
                    pass

            # Save run id and model info
            results['mlflow_run_id'] = run.info.run_id

    return results
