# Converted from Jupyter notebook using notebook-to-python converter
# Original notebook: cal_housing.ipynb

# ============================================================
# MARKDOWN CELL
# ============================================================
# # California Housing Price Prediction Project
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# ## 1. Data Preprocessing
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 1.1 Data Cleaning
# ============================================================

print("Missing values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 1.2 Data Exploration
# ============================================================

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['Target'], bins=30, kde=True)
plt.title("Distribution of Target (Median House Value)")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(15, 10))
df.boxplot(rot=45)
plt.title("Boxplot of Features to Identify Outliers")
plt.xticks(rotation=45)
plt.show()

num_features = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(15, 12))
for i, col in enumerate(num_features, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 1.3 Data Transformation
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# One-Hot Encoding for Linear Regression
# ============================================================

categorical_cols = df.select_dtypes(include=['object']).columns
df_ohe = df.copy()
df_ohe = pd.get_dummies(df_ohe, columns=categorical_cols, drop_first=True)

# ============================================================
# MARKDOWN CELL
# ============================================================
# Label Encoding for Tree-Based Models
# ============================================================

df_label = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_label[col] = le.fit_transform(df_label[col])
    label_encoders[col] = le

# ============================================================
# MARKDOWN CELL
# ============================================================
# Standardising the dataset
# ============================================================

scaler = StandardScaler()

df_label_scaled = df_label.copy()
df_label_scaled[df_label.columns.difference(categorical_cols)] = scaler.fit_transform(
    df_label[df_label.columns.difference(categorical_cols)]
)

df_ohe_scaled = pd.DataFrame(scaler.fit_transform(df_ohe), columns=df_ohe.columns)

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 1.4 Data Separation
# ============================================================

def split_data(df):
    X = df.drop(columns=["Target"])
    y = df["Target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = split_data(df_ohe_scaled)  #linear regression
X_train_label, X_test_label, y_train_label, y_test_label = split_data(df_label_scaled)  #tree based

# ============================================================
# MARKDOWN CELL
# ============================================================
# ## 2. Modeling
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 2.1 Model selection
# ============================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 2.2 Model training
# ============================================================

trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")

    if name == "Linear Regression":
        X_train, X_test, y_train, y_test = X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe
    else:
        X_train, X_test, y_train, y_test = X_train_label, X_test_label, y_train_label, y_test_label

    model.fit(X_train, y_train)
    trained_models[name] = model

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 2.3 Model optimisation
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# tuning hyperparameters using GridSearchCV for the Decision Tree, Random Forest, and Gradient Boosting models.
# ============================================================

param_grid = {
    "Decision Tree": {"max_depth": list(range(1, 21)) },
    "Random Forest": {"n_estimators": list(range(10, 210, 10)), "max_depth": list(range(1, 21))},
    "Gradient Boosting": {"n_estimators": list(range(10, 210, 10)), "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}
}

best_models = {}

for name, model in models.items():
    if name == "Linear Regression":
        best_models[name] = model
        continue #skip linear regression

    print(f"Tuning hyperparameters for {name}...")

    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train_label, y_train_label)

    best_models[name] = grid_search.best_estimator_
    print(f"Best params for {name}: {grid_search.best_params_}")

# ============================================================
# MARKDOWN CELL
# ============================================================
# ## 3. Model Evaluation
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 3.1 Evaluation metrics
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# To compare model performance, we will calculate:
#
# - **Mean Squared Error (MSE)**: Measures the average squared difference between predictions and actual values.
# - **Root Mean Squared Error (RMSE)**: The square root of MSE, making it more interpretable.
# - **Mean Absolute Error (MAE)**: Measures the average absolute error.
# - **R² Score**: Explains the variance in the target variable explained by the model.
# ============================================================

results = {}

for name, model in best_models.items():
    print(f"Evaluating {name}...")

    if name == "Linear Regression":
        X_test, y_test = X_test_ohe, y_test_ohe
    else:
        X_test, y_test = X_test_label, y_test_label

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

results_df = pd.DataFrame(results).T
print(results_df)

# ============================================================
# MARKDOWN CELL
# ============================================================
# ### 3.2 Visualisation
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# - **Residual plots**: To check for patterns in errors.
# - **Actual vs. Predicted plots**: To compare model predictions against actual values.
# ============================================================

plt.figure(figsize=(12, 6))

for i, (name, model) in enumerate(best_models.items()):
    plt.subplot(2, 2, i + 1)

    if name == "Linear Regression":
        X_test, y_test = X_test_ohe, y_test_ohe
    else:
        X_test, y_test = X_test_label, y_test_label

    y_pred = model.predict(X_test)

    sns.scatterplot(x=y_test, y=y_pred - y_test, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Actual Values ({name})")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

for i, (name, model) in enumerate(best_models.items()):
    plt.subplot(2, 2, i + 1)

    if name == "Linear Regression":
        X_test, y_test = X_test_ohe, y_test_ohe
    else:
        X_test, y_test = X_test_label, y_test_label

    y_pred = model.predict(X_test)

    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values ({name})")

plt.tight_layout()
plt.show()

# ============================================================
# MARKDOWN CELL
# ============================================================
# ## 4. Results analysis
# ============================================================

# ============================================================
# MARKDOWN CELL
# ============================================================
# Based on our evaluation metrics:
#
# - Which model performed best?
# - Why did this model outperform others?
# - What are the trade-offs between models?
# ============================================================

print("Final Model Performance Comparison:")
display(results_df)

best_model_name = results_df["R²"].idxmax()
print(f"\n🏆 The best performing model is: {best_model_name} with an R² score of {results_df.loc[best_model_name, 'R²']:.4f}")