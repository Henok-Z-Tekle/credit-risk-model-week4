
## Credit Risk Modeling – 10 Academy (Week 4)

This repository contains code, notebooks, and scripts for a full credit risk modeling pipeline, including EDA, feature engineering, model training, evaluation, and API deployment.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Task Breakdown](#task-breakdown)
- [EDA & Data Understanding](#eda--data-understanding)
- [Feature Engineering](#feature-engineering)
- [Model Training & Evaluation](#model-training--evaluation)
- [MLflow Integration](#mlflow-integration)
- [API Deployment](#api-deployment)
- [Quick Setup & Testing](#quick-setup--testing)
- [Project Structure](#project-structure)
- [References](#references)

---

## Project Overview
This project demonstrates a production-ready credit risk scoring pipeline using transactional data. It covers:
- Exploratory Data Analysis (EDA)
- Proxy target engineering
- Feature engineering (including RFM, WoE, etc.)
- Model training, evaluation, and selection
- MLflow experiment tracking and model registry
- FastAPI-based model serving

---

## Task Breakdown
1. **Business Understanding**: Framing credit risk, regulatory context, and proxy target design.
2. **EDA**: Data structure, missingness, distributions, outliers, and correlation analysis.
3. **Feature Engineering**: RFM, WoE, and custom features for credit risk.
4. **Model Training**: Train/test split, candidate models, metrics (AUC, accuracy, precision, recall, F1, confusion matrix), model selection, retraining, and persistence.
5. **MLflow Integration**: Experiment tracking, metrics logging, and model registry.
6. **API Deployment**: FastAPI app for prediction and model info, supports MLflow and local model loading.
7. **Testing & CI**: Unit tests, Docker, and CI workflow for reproducibility.

---

## EDA & Data Understanding
- Data loaded from `data/raw/` (see `notebooks/eda.ipynb`).
- Key steps: shape, types, summary stats, missing value analysis, outlier detection, categorical analysis, correlation matrix.
- Insights: Skewed monetary features, outliers, category concentration, correlated variables, and missing data strategies.

---

## Feature Engineering
- RFM (Recency, Frequency, Monetary) features
- Weight of Evidence (WoE) encoding
- Custom domain features
- See `src/processing/feature_engineering.py`, `src/processing/rfm.py`, `src/processing/woe.py`

---

## Model Training & Evaluation
- Models: Logistic Regression, Random Forest (see `src/models/train.py`)
- Stratified train/test split (if binary target)
- Metrics: AUC, accuracy, precision, recall, F1, confusion matrix
- Best model selected by test AUC, retrained on full data, saved to `models/model_best.joblib`
- MLflow logging: metrics, params, and model registry (optional)

---

## MLflow Integration
- Set `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT` to log runs
- Set `MLFLOW_REGISTER_MODEL` to register model in MLflow Model Registry
- Example (PowerShell):
```powershell
$env:MLFLOW_TRACKING_URI = 'http://localhost:5000'
$env:MLFLOW_EXPERIMENT = 'credit-risk'
# optionally: $env:MLFLOW_REGISTER_MODEL = 'credit-risk-model'
# from src.models.train import train_models
# results = train_models(X, y)
```

---

## API Deployment
- FastAPI app in `src/api/app.py`
- Loads model from MLflow (`MLFLOW_MODEL_URI`) or local path (`MODEL_PATH`)
- Endpoints:
  - `POST /predict` — returns `probability` and `prediction`
  - `GET /model-info` — returns model source (MLflow URI or local path)
- Example (PowerShell):
```powershell
$env:MODEL_PATH = 'models/model_best.joblib'
# or
$env:MLFLOW_MODEL_URI = 'models:/credit-risk-model/1'
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## Quick Setup & Testing
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

---

## Project Structure
```
├── data/
│   ├── raw/              # Original data files
│   └── processed/        # Cleaned/feature data
├── notebooks/            # EDA and analysis notebooks
├── src/
│   ├── api/              # FastAPI app
│   ├── eda/              # EDA scripts
│   ├── models/           # Model training code
│   ├── processing/       # Feature engineering
│   └── utils/            # Utility functions
├── models_test/          # Saved models
├── tests/                # Unit tests
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── README.md
```

---

## References
- [MLflow Documentation](https://mlflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [10 Academy](https://www.10academy.org/)
