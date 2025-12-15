# Task 3 — Feature Engineering

This folder implements Task 3 requirements: aggregate features, temporal extraction, encoding, scaling, and a simple WoE encoder.

Files added:
- `src/processing/feature_engineering.py` — aggregation and pipeline builder
- `src/processing/woe.py` — simple WoE encoder (fit/transform)
- `tests/test_feature_engineering.py` — unit tests validating aggregation and WoE

How to run locally:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_feature_engineering.py -q
```

Notes:
- The `create_customer_features` function aggregates transaction-level data into customer-level features required for modeling.
- The `build_feature_pipeline` returns an sklearn `Pipeline` to encode categorical variables and scale numeric features; it expects a customer-level DataFrame as input.
