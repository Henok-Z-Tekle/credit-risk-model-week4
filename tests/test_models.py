import pandas as pd
import numpy as np
from src.models import train


def make_sample_features(n=100):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        'recency_days': rng.randint(0, 100, size=n),
        'frequency': rng.randint(1, 10, size=n),
        'monetary': rng.uniform(1.0, 500.0, size=n),
    })
    # synthetic target correlated with monetary and frequency
    prob = (0.3 * (X['monetary'] / X['monetary'].max()) + 0.7 * (X['frequency'] / X['frequency'].max()))
    y = (prob + rng.normal(scale=0.05, size=n)) > 0.5
    return X, pd.Series(y.astype(int))


def test_train_models_runs():
    X, y = make_sample_features(60)
    res = train.train_models(X, y, output_dir='models_test')
    assert 'logistic' in res and 'random_forest' in res
    assert 'best' in res and 'path' in res['best']
