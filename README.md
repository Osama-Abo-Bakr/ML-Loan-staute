# Loan Status Prediction

## Project Overview

This project aims to predict the status of loans (approved or not approved) using various machine learning models. The project workflow includes data preprocessing, visualization, model development, and hyperparameter tuning.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Modeling](#modeling)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Conclusion](#conclusion)
11. [Contact](#contact)

## Introduction

Loan status prediction is crucial for financial institutions to assess the risk associated with loan applicants. This project leverages machine learning techniques to predict loan approval statuses based on applicant information.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **XGBoost**: Extreme Gradient Boosting

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Cleaning Data**:
   - Dropped the `Loan_ID` column.
   - Filled missing values in numerical columns with the mean and in categorical columns with the mode.
   - Converted categorical variables to numerical using `LabelEncoder`.

3. **Feature Engineering**:
   - Created a new feature by encoding `Loan_Status` and transforming categorical features.

## Exploratory Data Analysis

1. **Descriptive Statistics**:
   - Displayed basic statistics using `data.describe()`.

2. **Visualizations**:
   - Used count plots to show the distribution of loan statuses by education and marital status.
   - Created a heatmap to visualize correlations between features.

## Modeling

1. **Logistic Regression**:
   - Configured with `max_iter=20000` and `C=10`.

2. **Random Forest Classifier**:
   - Configured with `max_depth=4`, `n_estimators=50`, and `min_samples_leaf=4`.

3. **AdaBoost Classifier**:
   - Base estimator: `DecisionTreeClassifier` with `max_depth=20`, `min_samples_split=5`, and `min_samples_leaf=6`.
   - Configured with `n_estimators=50` and `learning_rate=0.2`.

4. **XGBoost Classifier**:
   - Configured with `n_estimators=10`, `max_depth=5`, `max_leaves=10`, `learning_rate=0.2`, `min_child_weight=15`, and `max_bin=5`.

## Hyperparameter Tuning

- **Random Forest Hyperparameter Tuning**:
  - Used `GridSearchCV` to find the best parameters for `RandomForestClassifier`.

```python
param = {
    "n_estimators": np.arange(100, 200, 10),
    "max_depth": np.arange(15, 25, 1),
    "min_samples_split": np.arange(1, 5)
}

new_model_random = GridSearchCV(
    estimator=model_RF, 
    param_grid=param,
    verbose=6,
    cv=5,
    n_jobs=-1
)
new_model_random.fit(x_train, y_train)
```

## Results

- **Logistic Regression**:
  - Training Accuracy: 0.814663951120163
  - Test Accuracy: 0.7886178861788617

- **Random Forest Classifier**:
  - Training Accuracy: 0.8167006109979633
  - Test Accuracy: 0.7886178861788617

- **AdaBoost Classifier**:
  - Training Accuracy: 1.0
  - Test Accuracy: 0.7560975609756098

- **XGBoost Classifier**:
  - Training Accuracy: 0.814663951120163
  - Test Accuracy: 0.7886178861788617

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/loan-status-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd loan-status-prediction
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict loan statuses on new data.

## Conclusion

This project demonstrates the use of various machine learning models to predict loan statuses. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into the factors affecting loan approval.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv("D:\Courses language programming\Machine Learning\Folder Machine Learning\Loan_staute\Loan_staute.csv")

# Data cleaning
data.drop(columns="Loan_ID", axis=1, inplace=True)
data = fillna_data(data)
data.replace({"Loan_Status": {"N": 0, "Y": 1}}, inplace=True)
data.replace(to_replace="3+", value=4, inplace=True)
data["Dependents"] = data["Dependents"].astype("int64")
data_object = data.select_dtypes(include=["object"])

la = LabelEncoder()
for col in data_object.columns:
    data[col] = la.fit_transform(data[col])

# Visualizations
plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, square=True, fmt="0.2f")
sns.countplot(x="Education", hue="Loan_Status", data=data)
sns.countplot(x="Married", hue="Loan_Status", data=data)

# Train-test split
X = data.drop(columns=["Loan_Status"], axis=1)
Y = data["Loan_Status"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42, shuffle=Y)

# Logistic Regression
model_log = LogisticRegression(max_iter=20000, C=10)
model_log.fit(x_train, y_train)
print(f"Logistic Regression - Train Accuracy: {model_log.score(x_train, y_train)}")
print(f"Logistic Regression - Test Accuracy: {model_log.score(x_test, y_test)}")

# Random Forest Classifier
model_RF = RandomForestClassifier(max_depth=4, n_estimators=50, min_samples_leaf=4)
model_RF.fit(x_train, y_train)
print(f"Random Forest - Train Accuracy: {model_RF.score(x_train, y_train)}")
print(f"Random Forest - Test Accuracy: {model_RF.score(x_test, y_test)}")

# Hyperparameter Tuning
param = {"n_estimators": np.arange(100, 200, 10), "max_depth": np.arange(15, 25, 1), "min_samples_split": np.arange(1, 5)}
new_model_random = GridSearchCV(estimator=model_RF, param_grid=param, verbose=6, cv=5, n_jobs=-1)
new_model_random.fit(x_train, y_train)
print(new_model_random.best_estimator_)

# AdaBoost Classifier
model_AD = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20, min_samples_split=5, min_samples_leaf=6, random_state=8), n_estimators=50, learning_rate=0.2)
model_AD.fit(x_train, y_train)
print(f"AdaBoost - Train Accuracy: {model_AD.score(x_train, y_train)}")
print(f"AdaBoost - Test Accuracy: {model_AD.score(x_test, y_test)}")

# XGBoost Classifier
model_xgb = xgb.XGBClassifier(n_estimators=10, max_depth=5, max_leaves=10, learning_rate=0.2, min_child_weight=15, max_bin=5)
model_xgb.fit(x_train, y_train)
print(f"XGBoost - Train Accuracy: {model_xgb.score(x_train, y_train)}")
print(f"XGBoost - Test Accuracy: {model_xgb.score(x_test, y_test)}")
```
