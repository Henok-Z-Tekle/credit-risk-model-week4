from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model_best.joblib')


class Features(BaseModel):
    recency_days: float
    frequency: float
    monetary: float


app = FastAPI()
_model = None


def load_model() -> Optional[object]:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    _model = joblib.load(MODEL_PATH)
    return _model


@app.get('/')
def root():
    return {'status': 'ok'}


@app.post('/predict')
def predict(features: Features):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')
    X = np.array([[features.recency_days, features.frequency, features.monetary]])
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(X)[:, 1][0])
    else:
        prob = float(model.predict(X)[0])
    pred = int(prob >= 0.5)
    return {'probability': prob, 'prediction': pred}
