from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import os
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import Features, PredictionResponse

try:
    import mlflow
    import mlflow.pyfunc
except Exception:
    mlflow = None

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model_best.joblib')
MLFLOW_MODEL_URI = os.environ.get('MLFLOW_MODEL_URI')





app = FastAPI()
_model = None


def load_model() -> Optional[object]:
    global _model
    if _model is not None:
        return _model
    # Prefer MLflow model if provided
    if MLFLOW_MODEL_URI and mlflow is not None:
        try:
            _model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
            _model.__loaded_from__ = f'mlflow:{MLFLOW_MODEL_URI}'
            return _model
        except Exception:
            # fall back to local model
            pass

    if not os.path.exists(MODEL_PATH):
        return None
    _model = joblib.load(MODEL_PATH)
    _model.__loaded_from__ = MODEL_PATH
    return _model


@app.get('/')
def root():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictionResponse)
def predict(features: Features):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')
    X = np.array([[features.recency_days, features.frequency, features.monetary]])
    # mlflow.pyfunc models expose a ``predict`` that returns probabilities
    try:
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X)[:, 1][0])
        else:
            pred = model.predict(X)
            # if predict returns probability-like float array
            if isinstance(pred, np.ndarray) and pred.dtype == float and pred.size == 1:
                prob = float(pred[0])
            else:
                prob = float(pred[0])
    except Exception:
        # last-resort: cast predict to float
        prob = float(model.predict(X)[0])
    pred = int(prob >= 0.5)
    return PredictionResponse(probability=prob, prediction=pred)


@app.get('/model-info')
def model_info():
    model = load_model()
    if model is None:
        raise HTTPException(status_code=404, detail='Model not available')
    source = getattr(model, '__loaded_from__', 'unknown')
    return {'model_source': source}
