
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

