import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import LinearRegressionGD

print("Loading data and model...")

# Load artifacts (including the 7 features list)
try:
    with open('model_artefacts.pkl', 'rb') as f:
        artefacts = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found. Please run 'python train.py' first.")
    exit()

model = artefacts['model']
x_mean = artefacts['x_mean']
x_std = artefacts['x_std']
features = artefacts['features']

# Load test data using all 7 features
test_df = pd.read_csv('test_data.csv')
X_test = test_df[features].values
y_test = test_df['Salary'].values

# Standardize the test data
X_test_scaled = (X_test - x_mean) / x_std

# Prediction & Evaluation
y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f"Test Set RMSE: ${rmse:,.2f}")

# Visualization for Multivariate Regression (True vs Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit Line')

plt.title('True Salary vs Predicted Salary (7 Features)')
plt.xlabel('True Salary ($)')
plt.ylabel('Predicted Salary ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('error_analysis.png')
print("New chart saved as 'error_analysis.png'")