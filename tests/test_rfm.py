from datetime import datetime

import pandas as pd
from src.processing import rfm


def make_sample_tx():
    data = [
        {'TransactionId': 't1', 'CustomerId': 'c1', 'TransactionStartTime': '2025-12-01', 'Amount': 100.0},
        {'TransactionId': 't2', 'CustomerId': 'c1', 'TransactionStartTime': '2025-12-05', 'Amount': 50.0},
        {'TransactionId': 't3', 'CustomerId': 'c2', 'TransactionStartTime': '2025-11-01', 'Amount': 10.0},
        {'TransactionId': 't4', 'CustomerId': 'c3', 'TransactionStartTime': '2025-10-01', 'Amount': 500.0},
        {'TransactionId': 't5', 'CustomerId': 'c3', 'TransactionStartTime': '2025-12-10', 'Amount': 200.0},
    ]
    return pd.DataFrame(data)


def test_compute_rfm_basic():
    tx = make_sample_tx()
    snapshot = pd.to_datetime('2025-12-15')
    r = rfm.compute_rfm(tx, snapshot_date=snapshot)
    # expected customers c1,c2,c3
    assert set(r['CustomerId']) == {'c1', 'c2', 'c3'}
    row_c1 = r[r['CustomerId'] == 'c1'].iloc[0]
    assert row_c1['frequency'] == 2
    assert row_c1['monetary'] == 150.0


def test_cluster_and_high_risk():
    tx = make_sample_tx()
    snapshot = pd.to_datetime('2025-12-15')
    r = rfm.compute_rfm(tx, snapshot_date=snapshot)
    clustered = rfm.cluster_customers_rfm(r, n_clusters=3, random_state=0)
    assert 'cluster' in clustered.columns
    labeled = rfm.assign_high_risk_label(clustered)
    assert 'is_high_risk' in labeled.columns
    # high risk cluster should be integer 0/1 and at least one row labeled
    assert labeled['is_high_risk'].isin([0, 1]).all()
    assert labeled['is_high_risk'].sum() >= 1
