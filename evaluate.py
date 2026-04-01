import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import LinearRegressionGD

print("model upload...")

# Test data charging
test_df = pd.read_csv('test_data.csv')
X_test = test_df[['Experience']].values
y_test = test_df['Salary'].values

# Model loading + standardisation parameters
with open('model_artefacts.pkl', 'rb') as f:
    artefacts = pickle.load(f)

model = artefacts['model']
x_mean = artefacts['x_mean']
x_std = artefacts['x_std']

# Data standardisation
X_test_scaled = (X_test - x_mean) / x_std

# Evaluation
y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f"MSE : {rmse:.2f} $")

# Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='true values', alpha=0.7)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Model Predictions')
plt.title('Salary Prediction vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True)
plt.savefig('error_analysis.png')
print("error_analysis.png saved.")