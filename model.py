import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent loop
        for _ in range(self.n_iterations):
            # Prediction
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate the derivatives
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Parameters update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Calculated the loss function (MSE)
            loss = (1 / n_samples) * np.sum((y_predicted - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias