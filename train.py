import numpy as np
import pandas as pd

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