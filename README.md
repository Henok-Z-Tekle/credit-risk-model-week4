
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
- Proxy target engineering and explicit integration into the training dataset
- Feature engineering (including RFM, WoE, etc.)
- Model training, evaluation, and selection
- MLflow experiment tracking and model registry
- FastAPI-based model serving

---


## Task Breakdown
1. **Business Understanding**: Framing credit risk, regulatory context, and proxy target design.
2. **EDA**: Data structure, missingness, distributions, outliers, and correlation analysis.
3. **Feature Engineering**: RFM, WoE, and custom features for credit risk.
4. **Target Integration**: Proxy target variable is explicitly created and merged into the training dataset before feature engineering and model training.
5. **Model Training**: Explicit train/test split, candidate models, metrics (AUC, accuracy, precision, recall, F1, confusion matrix, classification report), model selection, retraining, and persistence.
6. **MLflow Integration**: Experiment tracking, metrics logging, and model registry.
7. **API Deployment**: FastAPI app for prediction and model info, supports MLflow and local model loading.
8. **Testing & CI**: Unit tests, Docker, and CI workflow for reproducibility.

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
- **Target Integration**: The proxy target is explicitly created and merged with the feature set before splitting and training.
- **Explicit Data Splitting**: Stratified train/test split (if binary target) is performed before any model fitting or evaluation.
- **Evaluation Metrics**: The following metrics are computed and logged for each candidate model:
  - AUC (Area Under ROC Curve)
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Full Classification Report (if applicable)
- **Model Selection**: Best model is selected by test AUC, retrained on full data, and saved to `models/model_best.joblib`.
- **MLflow Logging**: All metrics, parameters, and model artifacts are logged to MLflow. Optionally, the model is registered in the MLflow Model Registry.

---


## MLflow Integration
- Set `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT` to log runs
- Set `MLFLOW_REGISTER_MODEL` to register model in MLflow Model Registry
- Example (PowerShell):
```powershell
$env:MLFLOW_TRACKING_URI = 'http://localhost:5000'
$env:MLFLOW_EXPERIMENT = 'credit-risk'
# optionally: $env:MLFLOW_REGISTER_MODEL = 'credit-risk-model'
# Ensure your proxy target is merged into the training set before calling train_models
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
- **Deployment Setup:**
  - To serve the API with local model:
    ```powershell
    $env:MODEL_PATH = 'models/model_best.joblib'
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    ```
  - To serve the API with a model from MLflow Model Registry:
    ```powershell
    $env:MLFLOW_MODEL_URI = 'models:/credit-risk-model/1'
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    ```
  - Ensure all environment variables are set before starting uvicorn. The API will auto-detect and load the model accordingly.

---


## Quick Setup & Testing
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```
*Before training, ensure your proxy target is created and merged with your features DataFrame.*

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
