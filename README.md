# Paymo💰 - Multi-Feature Salary Predictor (Gradient Descent From Scratch)

## Project Overview
This project implements a **Multivariate Linear Regression model from scratch** using Gradient Descent, relying only on `numpy` for core mathematical operations. It predicts an employee's salary based on 7 highly intuitive and practical features:

1. Years of Experience  
2. Education Level  
3. Team Size Managed  
4. Company Size (Total Employees)  
5. Professional Certifications  
6. Foreign Languages Spoken  
7. Working Hours per Week  

This repository was built to demonstrate a deep understanding of core Machine Learning mechanics, including:

- Vectorized gradient descent calculations  
- Feature standardization (Z-score normalization) and its impact on convergence  
- Strict train/test data isolation  
- MLOps best practices (saving/loading model artifacts via `pickle`)  

---

## Features

- **Custom Algorithm:** The linear regression model (`model.py`) is written entirely from scratch without using `scikit-learn` for the training loop.  
- **Realistic Synthetic Data:** Generates a dataset of 1,000 samples with 7 user-friendly features and irreducible noise.  
- **Pipeline Separation:** Training, evaluation, and inference are separated into dedicated scripts.  
- **Interactive CLI:** A command-line interface allowing users to input simple profile details and get real-time salary predictions.  

---

## Project Structure

- `model.py` : Contains the `LinearRegressionGD` class (loss calculation and weight updates).  
- `train.py` : Generates data, splits it, standardizes it, trains the model, and exports artifacts.  
- `evaluate.py` : Loads the test set and artifacts, calculates RMSE, and generates a visualization plot.  
- `interactive.py` : Interactive CLI tool for real-time predictions.  
- `requirements.txt` : Project dependencies.  

---

## How to Install

### 1. Clone the repository

```bash
git clone https://github.com/Zineeddine-Abd/Paymo.git
cd Paymo
```

---

### 2. Install dependencies

Make sure you have Python installed. Then run:

```bash
pip install -r requirements.txt
```

---

## 💻 How to Use

Run the scripts in the following order:

### 1. Train the model

```bash
python train.py
```

This generates:
- `model_artefacts.pkl`
- `test_data.csv`

---

### 2. Evaluate the model

```bash
python evaluate.py
```

Outputs:
- RMSE score  
- `error_analysis.png`

---

### 3. Run interactive predictor

```bash
python interactive.py
```

---

## 🧮 Mathematical Background

The model minimizes the Mean Squared Error (MSE) loss function:

$$
J(w, b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

Gradients used for updates:

$$
dw = \frac{1}{N} X^T (\hat{y} - y)
$$

$$
db = \frac{1}{N} \sum (\hat{y} - y)
$$