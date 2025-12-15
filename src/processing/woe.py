from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Simple Weight of Evidence encoder for categorical variables.

    Usage:
      enc = WoEEncoder()
      enc.fit(X['cat_col'], y)
      X_transformed = enc.transform(X['cat_col'])
    """
    def __init__(self):
        self.woe_map: Dict[str, float] = {}

    def fit(self, X, y):
        s = pd.Series(X).astype(object)
        y = pd.Series(y)
        df = pd.DataFrame({'x': s, 'y': y})
        grouped = df.groupby('x')['y'].agg(['sum', 'count'])
        grouped = grouped.rename(columns={'sum': 'pos', 'count': 'total'})
        grouped['neg'] = grouped['total'] - grouped['pos']
        # avoid division by zero
        eps = 0.5
        pos_total = max(df['y'].sum(), eps)
        neg_total = max((1 - df['y']).sum(), eps)
        grouped['pos_rate'] = (grouped['pos'] + eps) / pos_total
        grouped['neg_rate'] = (grouped['neg'] + eps) / neg_total
        grouped['woe'] = np.log(grouped['pos_rate'] / grouped['neg_rate'])
        self.woe_map = grouped['woe'].to_dict()
        self.default_woe = float(grouped['woe'].mean()) if not grouped.empty else 0.0
        return self

    def transform(self, X):
        s = pd.Series(X).astype(object)
        return s.map(self.woe_map).fillna(self.default_woe).to_numpy().reshape(-1, 1)
