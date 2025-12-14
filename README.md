
<<<<<<< HEAD
# Credit Risk Modeling – Week 4 (10 Academy)
## Task 2 — Exploratory Data Analysis (EDA)

### Objective
The objective of Task 2 is to explore and understand the structure, quality, and patterns in the credit risk dataset before any modeling is performed. This task focuses on identifying data distributions, missing values, correlations, and outliers that may influence credit risk modeling decisions.

---

### Dataset Overview
- **Source:** Transactional dataset provided for Week 4 Credit Risk Modeling
- **Storage:** Stored locally under `data/raw/` and excluded from version control
- **Observation Unit:** Individual transaction records
- **Feature Types:**
  - Numerical features (e.g., transaction amount, value)
  - Categorical features (e.g., ProductCategory, ChannelId, ProviderId)
  - Identifier and timestamp fields
=======
# Credit Risk Modeling – Week 4 (10 Academy) Task 1 - Understanding Credit Risk and Task 2 — Exploratory Data Analysis (EDA)

## Task 1 - Understanding Credit Risk
This project focuses on building a credit risk modeling pipeline using transactional data.  
The objective is to understand customer risk behavior, prepare the data for modeling, and build interpretable and effective credit scoring models in line with regulatory and business requirements.

## Credit Scoring Business Understanding

### 1. Credit Risk in Financial Institutions
Credit risk refers to the possibility that a borrower or transaction counterparty will fail to meet their financial obligations. For financial institutions, effective credit risk assessment is essential to maintaining portfolio stability, ensuring regulatory compliance, and protecting profitability.

In this project, credit risk modeling is used to identify high-risk transactions or customers using historical transactional data. The resulting risk scores can support better lending decisions, fraud prevention, and customer segmentation.
>>>>>>> main

---

### EDA Components Implemented

#### 1. Data Structure and Summary Statistics
- Dataset shape and feature list reviewed
- Data types inspected using `info()`
- Descriptive statistics computed for numerical and categorical features
- Central tendency, dispersion, and distribution characteristics analyzed

#### 2. Missing Value Analysis
- Missing values identified per column
- Missingness documented and reviewed
- Proposed imputation strategies:
  - Numerical variables: median imputation
  - Categorical variables: most frequent category or "Unknown"
  - Datetime fields: parsing with error handling

#### 3. Numerical Feature Distributions
- Histograms with KDE plotted for numerical variables
- Distribution skewness and heavy-tailed behavior identified
- Patterns indicating transaction concentration and variability discussed

#### 4. Outlier Detection
- Box plots used to detect extreme values in numerical features
- Outliers identified in monetary variables
- Implications for robust scaling and preprocessing noted

#### 5. Categorical Feature Analysis
- Frequency distributions analyzed for categorical features
- Dominant categories identified
- Variability across products, channels, and providers explored

#### 6. Correlation Analysis
- Correlation matrix computed for numerical features
- Heatmap visualization used to assess relationships
- Highly correlated variables identified for potential feature selection

---

### Key Insights from EDA
1. **Skewed Monetary Distributions:** Transaction-related numerical features are highly skewed, indicating many small transactions and a few extreme values.
2. **Presence of Outliers:** Significant outliers exist in monetary columns, which may distort model training if not handled properly.
3. **Category Concentration:** A small number of categories dominate some categorical features, suggesting behavioral clustering.
4. **Feature Relationships:** Certain numerical features exhibit strong correlations, indicating potential redundancy.
5. **Data Quality Considerations:** Missing values in critical fields must be carefully addressed to prevent bias or data leakage.

---

### Tools and Libraries Used
- **Pandas & NumPy:** Data manipulation and analysis
- **Matplotlib & Seaborn:** Visualization
- **SciPy / Scikit-learn:** Statistical support
- **Jupyter Notebook:** Interactive analysis environment

---

### Notebook Location
All exploratory analysis for Task 2 is implemented in:

<<<<<<< HEAD
=======
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

   ## Task 2 — Exploratory Data Analysis (EDA)

### Objective
The objective of Task 2 is to explore and understand the structure, quality, and patterns in the credit risk dataset before any modeling is performed. This task focuses on identifying data distributions, missing values, correlations, and outliers that may influence credit risk modeling decisions.

---

### Dataset Overview
- **Source:** Transactional dataset provided for Week 4 Credit Risk Modeling
- **Storage:** Stored locally under `data/raw/` and excluded from version control
- **Observation Unit:** Individual transaction records
- **Feature Types:**
  - Numerical features (e.g., transaction amount, value)
  - Categorical features (e.g., ProductCategory, ChannelId, ProviderId)
  - Identifier and timestamp fields

---

### EDA Components Implemented

#### 1. Data Structure and Summary Statistics
- Dataset shape and feature list reviewed
- Data types inspected using `info()`
- Descriptive statistics computed for numerical and categorical features
- Central tendency, dispersion, and distribution characteristics analyzed

#### 2. Missing Value Analysis
- Missing values identified per column
- Missingness documented and reviewed
- Proposed imputation strategies:
  - Numerical variables: median imputation
  - Categorical variables: most frequent category or "Unknown"
  - Datetime fields: parsing with error handling

#### 3. Numerical Feature Distributions
- Histograms with KDE plotted for numerical variables
- Distribution skewness and heavy-tailed behavior identified
- Patterns indicating transaction concentration and variability discussed

#### 4. Outlier Detection
- Box plots used to detect extreme values in numerical features
- Outliers identified in monetary variables
- Implications for robust scaling and preprocessing noted

#### 5. Categorical Feature Analysis
- Frequency distributions analyzed for categorical features
- Dominant categories identified
- Variability across products, channels, and providers explored

#### 6. Correlation Analysis
- Correlation matrix computed for numerical features
- Heatmap visualization used to assess relationships
- Highly correlated variables identified for potential feature selection

---

### Key Insights from EDA
1. **Skewed Monetary Distributions:** Transaction-related numerical features are highly skewed, indicating many small transactions and a few extreme values.
2. **Presence of Outliers:** Significant outliers exist in monetary columns, which may distort model training if not handled properly.
3. **Category Concentration:** A small number of categories dominate some categorical features, suggesting behavioral clustering.
4. **Feature Relationships:** Certain numerical features exhibit strong correlations, indicating potential redundancy.
5. **Data Quality Considerations:** Missing values in critical fields must be carefully addressed to prevent bias or data leakage.

---

### Tools and Libraries Used
- **Pandas & NumPy:** Data manipulation and analysis
- **Matplotlib & Seaborn:** Visualization
- **SciPy / Scikit-learn:** Statistical support
- **Jupyter Notebook:** Interactive analysis environment

---

### Notebook Location
All exploratory analysis for Task 2 is implemented in:
- `notebooks/eda.ipynb`

---

**Setup & Quick Commands**

- Create and activate virtual environment (Windows PowerShell):

	```powershell
	python -m venv venv
	.\venv\Scripts\Activate.ps1
	pip install --upgrade pip
	pip install -r requirements.txt
	```

- Run unit tests locally:

	```powershell
	python -m pytest -q
	```

- Run EDA notebook (from repository root):

	```powershell
	jupyter notebook notebooks/eda.ipynb
	```

**CI & Containerization**

- A minimal GitHub Actions workflow is provided at `.github/workflows/ci.yml` which installs dependencies and runs `pytest` on pushes and pull requests.
- A `Dockerfile` and `docker-compose.yml` are included for reproducible test runs and simple containerized workflows.

**Notes on feedback and next steps**

- The EDA notebook now uses `src.utils.load_csv` helper for consistent loading and the `src` package has been made importable so tests can run in CI.
- To improve deployment-readiness further, consider adding a small `api/` FastAPI app, model training scripts in `src/`, and a requirements lock (`requirements-lock.txt`) for deterministic builds.


>>>>>>> main
