from pydantic import BaseModel

class Features(BaseModel):
    recency_days: float
    frequency: float
    monetary: float

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
