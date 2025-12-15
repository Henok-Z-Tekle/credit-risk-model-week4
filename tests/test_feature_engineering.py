import pandas as pd
from src.processing.feature_engineering import create_customer_features, build_feature_pipeline
from src.processing.woe import WoEEncoder
import numpy as np


def make_sample_transactions():
    return pd.DataFrame(
        {
            'TransactionId': ['t1', 't2', 't3', 't4'],
            'CustomerId': ['c1', 'c1', 'c2', 'c3'],
            'Amount': [100.0, 50.0, 20.0, 5.0],
            'Value': [100.0, 50.0, 20.0, 5.0],
            'TransactionStartTime': ['2020-01-01T10:00:00Z', '2020-01-02T11:00:00Z', '2020-01-03T12:00:00Z', '2020-01-04T13:00:00Z'],
        }
    )


def test_create_customer_features_basic():
    df = make_sample_transactions()
    customers = create_customer_features(df)
    assert 'CustomerId' in customers.columns
    assert 'total_amount' in customers.columns
    row_c1 = customers[customers['CustomerId'] == 'c1'].iloc[0]
    assert row_c1['total_amount'] == 150.0
    assert row_c1['txn_count'] == 2


def test_woe_encoder():
    X = pd.Series(['a', 'a', 'b', 'c', 'b'])
    y = pd.Series([1, 0, 0, 1, 0])
    enc = WoEEncoder()
    enc.fit(X, y)
    transformed = enc.transform(pd.Series(['a', 'b', 'd']))
    # should return 3 rows
    assert transformed.shape[0] == 3
