"""
Example: Integrate is_high_risk target into processed features for model training.
"""
import pandas as pd
from src.processing.rfm import compute_rfm, cluster_customers_rfm, assign_high_risk_label
from src.processing.feature_engineering import create_customer_features

# Load your transaction data
transactions = pd.read_csv('data/raw/data.csv')

# Compute RFM
rfm_df = compute_rfm(transactions)
clustered = cluster_customers_rfm(rfm_df)
labeled = assign_high_risk_label(clustered)

# Create features
features = create_customer_features(transactions)

# Merge is_high_risk into features
features = features.merge(labeled[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
features['is_high_risk'] = features['is_high_risk'].fillna(0).astype(int)

# Now features can be used for model training:
# X = features.drop(['CustomerId', 'is_high_risk'], axis=1)
# y = features['is_high_risk']
