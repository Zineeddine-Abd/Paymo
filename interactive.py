import numpy as np
import pickle
from model import LinearRegressionGD

# Load the model and standardization parameters
try:
    with open('model_artefacts.pkl', 'rb') as f:
        artefacts = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found. Please run 'python train.py' first.")
    exit()

model = artefacts['model']
x_mean = artefacts['x_mean']
x_std = artefacts['x_std']

print("\n======================================")
print(" MULTI-FEATURE SALARY PREDICTOR")
print("========================================")

while True:
    print("\n--- Enter profile details ---")
    try:
        exp = float(input("1. Years of experience (e.g : 5): "))
        edu = float(input("2. Years of education (e.g : 17 for Masters): "))
        mgt = float(input("3. Management position? (0 = No, 1 = Yes): "))
        size = float(input("4. Company size (1=Small, 2=Medium, 3=Large): "))
        loc = float(input("5. Location index (e.g : 1.0 standard, 1.5 high-cost): "))
        perf = float(input("6. Performance rating (1 to 5): "))
        hrs = float(input("7. Work hours per week (e.g : 35): "))
        
        # Format inputs into a NumPy array
        user_input = np.array([[exp, edu, mgt, size, loc, perf, hrs]])
        
        # Standardize the input using the Train Set parameters
        user_input_scaled = (user_input - x_mean) / x_std
        
        # Prediction
        prediction = model.predict(user_input_scaled)[0]
        
        print("\n" + "="*42)
        print(f" ESTIMATED SALARY : ${prediction:,.2f}")
        print("="*42)
        
    except ValueError:
        print("\nError: Please enter valid numbers only.")
    
    # Ask if the user wants to continue
    continue_prompt = input("\nDo you want to test another profile? (y/n): ").strip().lower()
    if continue_prompt != 'y':
        print("Thank you for using the predictor. Goodbye!")
        break