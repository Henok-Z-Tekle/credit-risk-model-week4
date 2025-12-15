API — Run instructions

Run locally (uvicorn):

1. Ensure a trained model exists at `models/model_best.joblib` (the training step saves the best model there).
2. Start the API:

```powershell
# from repository root
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

3. Query the predict endpoint:

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"recency_days":5, "frequency":2, "monetary":50}'
```

Run via Docker:

```powershell
docker build -f Dockerfile.api -t credit-api .
docker run -p 8000:8000 credit-api
```

Notes:
- The service expects environment variable `MODEL_PATH` pointing to the joblib model file (default: `/app/models/model_best.joblib`).
- CI is configured to start a local MLflow server and tests will log runs during the test job.
