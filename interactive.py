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

print("\n==========================================")
print(" 💰 EASY SALARY PREDICTOR TOOL")
print("==========================================")

while True:
    print("\n--- Please answer 7 simple questions ---")
    try:
        exp = float(input("1. How many years of professional experience do you have? (e.g., 5): "))
        
        print("\n   [1] High School   [2] Bachelor's   [3] Master's   [4] PhD")
        edu = float(input("2. What is your highest education level? (Enter 1, 2, 3, or 4): "))
        
        team = float(input("3. How many people do you directly manage? (Enter 0 if none): "))
        
        comp = float(input("4. How many employees work at your company? (e.g : 50, 1000): "))
        
        cert = float(input("5. How many professional certifications do you hold? (e.g : 0, 1, 2): "))
        
        lang = float(input("6. How many foreign languages do you speak? (e.g : 0, 1, 2): "))
        
        hrs = float(input("7. How many hours do you work per week? (e.g : 35, 40): "))
        
        # Format inputs into a NumPy array
        user_input = np.array([[exp, edu, team, comp, cert, lang, hrs]])
        
        # Standardize the input using the Train Set parameters
        user_input_scaled = (user_input - x_mean) / x_std
        
        # Prediction
        prediction = model.predict(user_input_scaled)[0]
        
        print("\n" + "="*45)
        print(f" YOUR ESTIMATED YEARLY SALARY : ${prediction:,.2f}")
        print("="*45)
        
    except ValueError:
        print("\nError: Please enter valid numbers only.")
    
    # Ask if the user wants to continue
    continue_prompt = input("\nDo you want to predict another salary? (y/n): ").strip().lower()
    if continue_prompt != 'y':
        print("Thank you for using the Paymo !")
        break