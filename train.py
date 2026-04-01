import numpy as np
import pandas as pd
from model import LinearRegressionGD

np.random.seed(42)
X = 5 * np.random.rand(100, 1) 
y = 40000 + 8000 * X + np.random.randn(100, 1) * 3000

df = pd.DataFrame({'Experience': X.flatten(), 'Salary': y.flatten()})
print("Data visualization :")
print(df.head())

# manual Split Train/Test (80/20)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_train = train_df[['Experience']].values
y_train = train_df['Salary'].values
X_test = test_df[['Experience']].values
y_test = test_df['Salary'].values

# Data Scaling (Standardization)
x_mean, x_std = X_train.mean(), X_train.std()
X_train_scaled = (X_train - x_mean) / x_std
X_test_scaled = (X_test - x_mean) / x_std

# Init
print("\nStarting training...")
model = LinearRegressionGD(learning_rate=0.1, n_iterations=200)
model.fit(X_train_scaled, y_train)

print(f"Training completed. Weights : {model.weights[0]:.2f}, Bias : {model.bias:.2f}")