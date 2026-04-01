# 💰 Multi-Feature Salary Predictor (Gradient Descent From Scratch)

## Project Overview
This project implements a **Multivariate Linear Regression model from scratch** using Gradient Descent, relying only on `numpy` for core mathematical operations. It predicts an employee's salary based on 7 distinct features (experience, education, management role, etc.).

This repository was built to demonstrate a deep understanding of core Machine Learning mechanics, including:
- Vectorized gradient descent calculations.
- Feature standardization (Z-score normalization) and its impact on convergence.
- Strict train/test data isolation.
- MLOps best practices (saving/loading model artifacts via `pickle`).

## Features
* **Custom Algorithm:** The linear regression model (`model.py`) is written entirely from scratch without using `scikit-learn` for the training loop.
* **Realistic Synthetic Data:** Generates a dataset of 1,000 samples with 7 features and irreducible noise.
* **Pipeline Separation:** Training, evaluation, and inference are separated into dedicated scripts.
* **Interactive CLI:** A command-line interface allowing users to input profile details and get real-time salary predictions.

## Project Structure
* `model.py` : Contains the `LinearRegressionGD` class (Loss calculation and weight updates).
* `train.py` : Generates data, splits it, standardizes it, trains the model, and exports artifacts.
* `evaluate.py` : Loads the test set and artifacts, calculates RMSE, and generates a visualization plot.
* `interactive.py` : An interactive CLI tool for real-time predictions on user-provided data.
* `requirements.txt` : Project dependencies.

## How to Install

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Zineeddine-Abd/Paymo.git](https://github.com/Zineeddine-Abd/Paymo.git)
   cd Paymo

## 📦 Install dependencies

Make sure you have Python installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

---

## 💻 How to Use

You must run the scripts in the following order to ensure the model artifacts are properly generated:

### 1. Train the Model

Run the training script to generate the synthetic dataset, train the model, and save the weights and scaling parameters:

```bash
python train.py
```

*(This will generate `model_artefacts.pkl` and `test_data.csv`)*

---

### 2. Evaluate the Model

Test the model's performance on the unseen test set:

```bash
python evaluate.py
```

*(This will print the RMSE and generate an `error_analysis.png` chart)*

---

### 3. Run the Interactive Predictor

Start the CLI tool to enter custom data and predict a salary:

```bash
python interactive.py
```

---

## 🧮 Mathematical Background

The model minimizes the Mean Squared Error (MSE) loss function:

$$
J(w, b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

Where the gradients used to update the weights and bias at each iteration are calculated as:

$$
dw = \frac{1}{N} X^T (\hat{y} - y)
$$

$$
db = \frac{1}{N} \sum (\hat{y} - y)
$$