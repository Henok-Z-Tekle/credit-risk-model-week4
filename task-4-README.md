Task 4 — RFM segmentation and high-risk labeling

Files added:

- `src/processing/rfm.py` — functions `compute_rfm`, `cluster_customers_rfm`, `assign_high_risk_label`.
- `tests/test_rfm.py` — unit tests for RFM and clustering.

How to run tests:

1. Activate the project's virtualenv.
2. Install requirements: `pip install -r requirements.txt` (ensure scikit-learn is installed).
3. Run pytest: `python -m pytest -q`
