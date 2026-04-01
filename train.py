import numpy as np
import pandas as pd
import pickle
from model import LinearRegressionGD

# Create the dataset (7 intuitive features, 1000 samples)
np.random.seed(42)
n_samples = 1000

# Generate features
years_experience = np.random.randint(0, 25, n_samples)
education_level = np.random.randint(1, 5, n_samples) # 1:High School, 2:Bachelor, 3:Master, 4:PhD
team_size = np.random.randint(0, 15, n_samples) # Number of people managed
company_employees = np.random.randint(10, 5000, n_samples) # Total company employees
certifications = np.random.randint(0, 6, n_samples) # Number of pro certificates
foreign_languages = np.random.randint(0, 4, n_samples) # Languages spoken
hours_per_week = np.random.randint(35, 55, n_samples)

# Salary formula based on these new features (with random noise)
salary = (
    30000 + 
    (years_experience * 2000) + 
    (education_level * 5000) + 
    (team_size * 800) + 
    (company_employees * 2) + 
    (certifications * 1500) + 
    (foreign_languages * 2000) + 
    (hours_per_week * 200) + 
    (np.random.randn(n_samples) * 3000) # Irreducible noise
)

df = pd.DataFrame({
    'Experience': years_experience, 
    'Education_Level': education_level, 
    'Team_Size': team_size, 
    'Company_Employees': company_employees, 
    'Certifications': certifications, 
    'Foreign_Languages': foreign_languages, 
    'Hours_Week': hours_per_week, 
    'Salary': salary
})

# Train/Test Split (800 / 200)
train_size = 800
train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

features = [
    'Experience', 'Education_Level', 'Team_Size', 'Company_Employees', 
    'Certifications', 'Foreign_Languages', 'Hours_Week'
]
X_train, y_train = train_df[features].values, train_df['Salary'].values

# Standardization (calculated for each feature independently)
x_mean = np.mean(X_train, axis=0)
x_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - x_mean) / x_std

# Training
print(f"Training on {train_size} samples and {len(features)} intuitive features...")
model = LinearRegressionGD(learning_rate=0.05, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Save artifacts
test_df.to_csv('test_data.csv', index=False)
artefacts = {'model': model, 'x_mean': x_mean, 'x_std': x_std, 'features': features}
with open('model_artefacts.pkl', 'wb') as f:
    pickle.dump(artefacts, f)

print("Model trained and saved successfully!")