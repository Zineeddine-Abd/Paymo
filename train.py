import numpy as np
import pandas as pd
import pickle
from model import LinearRegressionGD

# Create the dataset (7 features, 1000 samples)
np.random.seed(42)
n_samples = 1000

# Generate the features
experience = np.random.randint(0, 20, n_samples)
education_years = np.random.randint(12, 22, n_samples) # 12 (High School) to 20 (PhD)
management = np.random.randint(0, 2, n_samples) # 0 (No) or 1 (Yes)
company_size = np.random.randint(1, 4, n_samples) # 1 (Small), 2 (Medium), 3 (Large)
location_index = np.random.uniform(0.8, 1.5, n_samples) # Cost of living (e.g : Paris = 1.5)
performance = np.random.randint(1, 6, n_samples) # Rating from 1 to 5
hours_per_week = np.random.randint(35, 55, n_samples)

# Salary formula (with random noise for realism)
salary = (
    30000 + 
    (experience * 2500) + 
    (education_years * 1200) + 
    (management * 8000) + 
    (company_size * 2000) + 
    (location_index * 6000) + 
    (performance * 1500) + 
    (hours_per_week * 150) + 
    (np.random.randn(n_samples) * 3500) # Noise (irreducible error)
)

df = pd.DataFrame({
    'Experience': experience, 'Education_Years': education_years, 
    'Management': management, 'Company_Size': company_size, 
    'Location_Index': location_index, 'Performance': performance, 
    'Hours_Week': hours_per_week, 'Salary': salary
})

# Train/Test Split (800 / 200)
train_size = 800
train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

features = ['Experience', 'Education_Years', 'Management', 'Company_Size', 'Location_Index', 'Performance', 'Hours_Week']
X_train, y_train = train_df[features].values, train_df['Salary'].values

# Standardization (calculated for each feature independently)
x_mean = np.mean(X_train, axis=0)
x_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - x_mean) / x_std

# Training
print(f"Training on {train_size} samples and {len(features)} features...")
model = LinearRegressionGD(learning_rate=0.05, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Save artifacts
test_df.to_csv('test_data.csv', index=False)
artefacts = {'model': model, 'x_mean': x_mean, 'x_std': x_std, 'features': features}
with open('model_artefacts.pkl', 'wb') as f:
    pickle.dump(artefacts, f)

print("Model trained and saved successfully!")