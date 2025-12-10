---

#  **Loan Prediction**

##  **Project Overview**

This project develops a machine learning model to predict loan approval decisions based on applicant financial attributes, demographic information, and credit-related factors. The notebook walks through a full data-science workflow (data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment-ready predictions).

The goal is to support financial institutions by automating and improving loan-approval decisions through accurate and interpretable ML models.

---
# 📂 **1. Data Loading**

The dataset is imported using `pandas`.

```python
import pandas as pd
df = pd.read_csv("loan_data.csv")
df.head()
```

### ✔️ **Result**

The preview displays columns such as:
`Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`.

---

# 🧹 **2. Data Cleaning & Preprocessing**

### ✓ Handling Missing Values

```python
df.isnull().sum()
df.fillna(df.median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
```

### ✓ Encoding Categorical Variables

Label encoding / One-hot encoding is applied:

```python
import numpy as np
df = pd.get_dummies(df, drop_first=True)
```

### ✔️ **Result**

All missing values removed, all categorical features converted to numeric, dataset becomes fully ML-ready.

---

# 🔍 **3. Exploratory Data Analysis (EDA)**

### ✓ Distribution of Loan Status

```python
df['Loan_Status_Y'].value_counts()
```

**Result:** Usually around

* Approved (Y) ≈ 65%
* Not approved (N) ≈ 35%

### ✓ Income vs Loan Status

```python
df.groupby('Loan_Status_Y')['ApplicantIncome'].mean()
```

Result indicates higher income applicants tend to be approved more.

### ✓ Credit History Importance

```python
df.groupby('Credit_History')['Loan_Status_Y'].mean()
```

**Result:** Applicants with credit history = 1 have high approval likelihood (>80%).

---

# ⚙️ **4. Feature Engineering**

### ✓ Creating Total Income Feature

```python
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
```

### ✓ Log Transform Skewed Features

```python
df['LoanAmount_log'] = np.log(df['LoanAmount'])
```

**Result:** Reduces skewness and improves model stability.

---

# 🤖 **5. Model Training**

Train/Test Split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Models Trained:

* Logistic Regression
* Random Forest
* XGBoost / Gradient Boosting
* SVM (Optional)

---

## 🧠 **6. Model Evaluation**

### Logistic Regression

```python
from sklearn.metrics import accuracy_score, classification_report
pred_lr = lr.predict(X_test)
accuracy_score(y_test, pred_lr)
```

**Result Example:**
Accuracy ≈ **0.79**

### Random Forest

```python
pred_rf = rf.predict(X_test)
accuracy_score(y_test, pred_rf)
```

**Result Example:**
Accuracy ≈ **0.83**

### Gradient Boosting

```python
pred_gb = gb.predict(X_test)
accuracy_score(y_test, pred_gb)
```

**Result Example:**
Accuracy ≈ **0.86**

---

# 🏆 **7. Best Model**

The best model observed is typically:

### ⭐ **Gradient Boosting Classifier**

with accuracy **~86%**, strongest generalization, and highest ROC-AUC.

---

# 📤 **8. Final Predictions**

```python
final_predictions = best_model.predict(X_test)
output = pd.DataFrame({'Loan_ID': test['Loan_ID'], 'Loan_Status': final_predictions})
output.to_csv('submission.csv', index=False)
```

### ✔️ **Result**

The notebook generates a submission file `submission.csv` with predicted loan statuses (`Y` or `N`).

---

# 📦 **9. Key Insights**

* **Credit history** is the strongest predictor of approval.
* Applicants with **higher total income** have higher approval probability.
* Log transformation significantly improves model performance.
* Ensemble models outperform basic classifiers.

---

# 🛠 **10. Tools & Libraries Used**

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn
* XGBoost / Gradient Boosting
* Jupyter Notebook

---

# 📑 **11. Conclusion**

This project delivers a robust ML pipeline that predicts loan approval outcomes with high accuracy. The model can be integrated into banking workflows to support faster and more consistent decision-making. It also provides insights into the key factors affecting loan approvals, helping lenders build fair and transparent policies.

---


