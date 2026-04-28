import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import seaborn as sns

FILE_PATH = "Concrete_Data.csv"
TARGET_COLUMN = "Concrete compressive strength(MPa, megapascals) " 
df = pd.read_csv(FILE_PATH)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

df = df.select_dtypes(include=[np.number])

print("\nSummary Statistics:\n", df.describe())

print("\nMissing Values:\n", df.isnull().sum())

df.hist(figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

corr = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
top_feature = corr.abs().idxmax()

plt.scatter(df[top_feature], df[TARGET_COLUMN])
plt.xlabel(top_feature)
plt.ylabel(TARGET_COLUMN)
plt.title(f"{top_feature} vs {TARGET_COLUMN}")
plt.show()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"])
print("\nModel Comparison:")
print(results_df)