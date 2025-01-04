import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def expand_features(X, degree):
    m, n = X.shape
    X_poly = np.ones((m, 1))
    for d in range(1, degree + 1):
        for feature in range(n):
            X_poly = np.hstack((X_poly, (X[:, feature]**d).reshape(-1, 1)))
    mean = np.mean(X_poly, axis=0)
    std = np.std(X_poly, axis=0)
    std[std == 0] = 1e-8
    X_poly = (X_poly - mean) / std
    return X_poly

def predict(X_poly, w, b):
    return np.dot(X_poly, w) + b

def compute_cost(y, y_pred):
    return np.mean((y - y_pred)**2)

def gradient_descent(X_poly, y, w, b, alpha):
    m = len(y)
    y_pred = predict(X_poly, w, b)
    error = y_pred - y
    dW = (1 / m) * np.dot(X_poly.T, error)
    dB = (1 / m) * np.sum(error)
    w -= alpha * dW
    b -= alpha * dB
    return w, b

def train(X, y, degree, alpha, iters):
    X_poly = expand_features(X, degree)
    n = X_poly.shape[1]
    w = np.zeros(n)
    b = 0
    cost_history = []
    for i in range(iters):
        w, b = gradient_descent(X_poly, y, w, b, alpha)
        y_pred = predict(X_poly, w, b)
        cost = compute_cost(y, y_pred)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Epoch {i}, Cost: {cost}")
    plt.plot(range(iters), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()
    return w, b

def evaluate(X, y, w, b, degree):
    X_poly = expand_features(X, degree)
    y_pred = predict(X_poly, w, b)
    mean_y = np.mean(y)
    ss_total = np.sum((y - mean_y)**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared, y_pred

df = pd.read_csv("/content/drive/MyDrive/Machine Learning Library /Datasets/polynomial_regression_train.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
plt.scatter(X_norm[:, 0], y, color='blue')
plt.xlabel("Feature 1 (Normalized)")
plt.ylabel("Target Variable")
plt.title("Original Data")
plt.show()

degree = 3
alpha = 0.01
iters = 5000
w, b = train(X_norm, y, degree, alpha, iters)
r2, y_pred = evaluate(X_norm, y, w, b, degree)
print(f"R-squared: {r2}")
plt.scatter(X_norm[:, 0], y, color='blue', label="Original Data")
plt.scatter(X_norm[:, 0], y_pred, color='red', label="Predicted Data")
plt.xlabel("Feature 1 (Normalized)")
plt.ylabel("Target Variable")
plt.title("Polynomial Regression Predictions")
plt.legend()
plt.show()
