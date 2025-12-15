from __future__ import annotations

from typing import Iterable
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to customer-level features.

    Produces:
      - total_amount: sum of Amount per customer
      - avg_amount: mean Amount per customer
      - txn_count: number of transactions per customer
      - std_amount: standard deviation of Amount per customer (0 if single tx)
      - last_tx_hour/day/month/year: extracted from the most recent TransactionStartTime

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level dataframe. Must contain `CustomerId`, `Amount`, and `TransactionStartTime`.

    Returns
    -------
    pd.DataFrame
        Customer-level features indexed by `CustomerId`.
    """
    if not {'CustomerId', 'Amount', 'TransactionStartTime'}.issubset(df.columns):
        raise ValueError('DataFrame must contain CustomerId, Amount and TransactionStartTime')

    # ensure time parsed
    times = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df = df.copy()
    df['__ts'] = times

    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        txn_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std'),
        last_ts=('__ts', 'max'),
    )

    agg['std_amount'] = agg['std_amount'].fillna(0.0)

    # temporal features from last transaction
    agg['last_tx_hour'] = agg['last_ts'].dt.hour.fillna(-1).astype(int)
    agg['last_tx_day'] = agg['last_ts'].dt.day.fillna(-1).astype(int)
    agg['last_tx_month'] = agg['last_ts'].dt.month.fillna(-1).astype(int)
    agg['last_tx_year'] = agg['last_ts'].dt.year.fillna(-1).astype(int)

    # drop helper
    agg = agg.drop(columns=['last_ts'])
    agg = agg.reset_index()
    return agg


def build_feature_pipeline(categorical_features: Iterable[str], numeric_features: Iterable[str]) -> Pipeline:
    """Return an sklearn Pipeline that encodes categoricals and scales numerics.

    This pipeline expects a customer-level DataFrame (one row per customer).
    """
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, list(numeric_features)),
            ('cat', cat_transformer, list(categorical_features)),
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
    ])

    return pipeline
