# credit-risk-model-week4

# credit-risk-model-week4

# Credit Risk Modeling – Week 4 (10 Academy)

This project focuses on building a credit risk modeling pipeline using transactional data.  
The objective is to understand customer risk behavior, prepare the data for modeling, and build interpretable and effective credit scoring models in line with regulatory and business requirements.

---

## Credit Scoring Business Understanding

### 1. Credit Risk in Financial Institutions
Credit risk refers to the possibility that a borrower or transaction counterparty will fail to meet their financial obligations. For financial institutions, effective credit risk assessment is essential to maintaining portfolio stability, ensuring regulatory compliance, and protecting profitability.

In this project, credit risk modeling is used to identify high-risk transactions or customers using historical transactional data. The resulting risk scores can support better lending decisions, fraud prevention, and customer segmentation.

---

### 2. Basel II Accord and the Importance of Interpretability
The Basel II Accord emphasizes risk-sensitive capital allocation and requires financial institutions to justify how risk is measured and managed. Under Basel II, models used for credit risk assessment must be:
- Transparent and interpretable
- Well-documented and auditable
- Consistent and reproducible

Because regulatory bodies may review model decisions, it is not sufficient for a model to be accurate. Stakeholders must be able to understand how predictions are made and which factors influence risk outcomes. This makes interpretability a critical requirement in regulated financial environments.

---

### 3. Proxy Variable for Default Risk
The dataset used in this project does not contain a direct label indicating whether a customer has defaulted. Since supervised machine learning models require labeled outcomes, a proxy variable must be created to represent credit risk.

The proxy variable is derived from observable behavioral or transactional patterns that are assumed to correlate with higher risk. While this enables model training, it introduces potential business risks, including:
- Misclassification of customers who behave unusually but are not truly risky
- Bias against specific customer groups
- Incorrect pricing or rejection of legitimate customers

These risks highlight the importance of carefully defining the proxy target and continuously monitoring model performance after deployment.

---

### 4. Trade-offs Between Model Complexity and Interpretability
Simple models such as Logistic Regression combined with Weight of Evidence (WoE) encoding are widely used in regulated credit environments. These models are easy to interpret and allow clear explanations of how each feature contributes to risk predictions.

More complex models, such as Gradient Boosting or XGBoost, often achieve higher predictive accuracy by capturing non-linear relationships and feature interactions. However, they are less transparent and require additional interpretability tools (e.g., SHAP) to explain their predictions.

In practice, financial institutions must balance predictive performance with regulatory requirements. A common strategy is to start with interpretable models for compliance and gradually incorporate more complex models where explainability can be adequately supported.
