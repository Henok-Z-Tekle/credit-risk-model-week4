from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_rfm(transactions: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary (RFM) per CustomerId.

    Parameters
    ----------
    transactions : pd.DataFrame
        Transaction-level dataframe containing `CustomerId`, `TransactionStartTime`, and `Amount`.
    snapshot_date : pd.Timestamp, optional
        Date to use as the reference for Recency calculation. If None, uses max TransactionStartTime.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by CustomerId with columns `recency_days`, `frequency`, `monetary`.
    """
    if 'CustomerId' not in transactions.columns:
        raise ValueError('transactions must include CustomerId')

    tx = transactions.copy()
    tx['ts'] = pd.to_datetime(tx['TransactionStartTime'], errors='coerce')
    if snapshot_date is None:
        snapshot = tx['ts'].max()
    else:
        snapshot = pd.to_datetime(snapshot_date)

    grouped = tx.groupby('CustomerId').agg(
        last_ts=('ts', 'max'),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
    )

    grouped['recency_days'] = (snapshot - grouped['last_ts']).dt.days.fillna(-1).astype(int)
    grouped = grouped[['recency_days', 'frequency', 'monetary']].reset_index()
    return grouped


def cluster_customers_rfm(rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Cluster customers using KMeans on RFM features.

    Returns the input DataFrame with an added `cluster` column.
    """
    if not {'recency_days', 'frequency', 'monetary', 'CustomerId'}.issubset(rfm_df.columns):
        raise ValueError('rfm_df must contain CustomerId, recency_days, frequency, monetary')

    X = rfm_df[['recency_days', 'frequency', 'monetary']].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(Xs)
    out = rfm_df.copy()
    out['cluster'] = labels
    return out


def assign_high_risk_label(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Assign is_high_risk = 1 for the cluster with lowest frequency+monetary.

    The heuristic: compute cluster-level mean of frequency and monetary, rank clusters by (frequency + monetary) ascending.
    The lowest-ranked cluster is considered high risk.
    """
    if 'cluster' not in clustered_df.columns:
        raise ValueError('clustered_df must have a cluster column')

    stats = clustered_df.groupby('cluster').agg(frequency_mean=('frequency', 'mean'), monetary_mean=('monetary', 'mean'))
    stats['score'] = stats['frequency_mean'] + stats['monetary_mean']
    high_risk_cluster = int(stats['score'].idxmin())
    out = clustered_df.copy()
    out['is_high_risk'] = (out['cluster'] == high_risk_cluster).astype(int)
    return out
