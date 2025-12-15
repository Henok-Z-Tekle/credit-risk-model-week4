import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient


def test_predict_endpoint(tmp_path, monkeypatch):
    # train a tiny model and save it to a temp models directory
    X = np.array([[0, 1, 10], [10, 2, 100]])
    y = np.array([1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    model_path = str(models_dir / 'model_best.joblib')
    joblib.dump(model, model_path)

    # set env var and reload app module
    monkeypatch.setenv('MODEL_PATH', model_path)
    import importlib
    import src.api.app as appmod
    importlib.reload(appmod)

    client = TestClient(appmod.app)
    resp = client.post('/predict', json={'recency_days': 5, 'frequency': 2, 'monetary': 50})
    assert resp.status_code == 200
    j = resp.json()
    assert 'probability' in j and 'prediction' in j
