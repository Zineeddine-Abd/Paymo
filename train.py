import numpy as np
import pandas as pd

np.random.seed(42)
X = 5 * np.random.rand(100, 1) 
y = 40000 + 8000 * X + np.random.randn(100, 1) * 3000

df = pd.DataFrame({'Experience': X.flatten(), 'Salary': y.flatten()})
print("Data visualization :")
print(df.head())
