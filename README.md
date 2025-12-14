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

---

### Rubric-Focused Answers (Task 1)

- **How Basel II's emphasis on risk measurement influences the need for interpretability:**
	- Basel II requires institutions to demonstrate how capital requirements are derived from measured risks. This creates a strong need for models that are interpretable, reproducible, and auditable so that regulators and internal stakeholders can validate the model logic and its impact on capital.

- **Why create a proxy variable for default (given no direct label):**
	- Because the dataset lacks an explicit default flag, a proxy variable (for example, prolonged delinquency, charge-offs, or sustained negative balances) is necessary to create a supervised learning target.
	- This proxy enables model training but must be carefully designed and validated to avoid introducing label noise or systemic bias.
	- Potential business risks include misclassification, unfair treatment of customer cohorts, and incorrect credit-decisioning that impacts revenue or compliance.

- **Key trade-offs between simple (Logistic Regression + WoE) and complex models (Gradient Boosting):**
	- *Interpretability vs Performance:* Logistic Regression with WoE is transparent and easy to explain; Gradient Boosting often provides better predictive power but needs explainability tools (SHAP) and more rigorous validation.
	- *Regulatory Acceptability:* Simpler models are easier to defend in regulated reviews; complex models require additional documentation, monitoring, and justification.
	- *Operational Complexity:* Complex models typically require more infrastructure, hyperparameter tuning, and monitoring for concept drift.
	- *Recommended approach:* Start with a well-documented, interpretable model as a baseline. If complex models are adopted, provide robust explanation tools, model cards, and monitoring to satisfy regulatory expectations.
