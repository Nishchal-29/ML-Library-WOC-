import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_and_normalize_dataset(train_filepath):
    df = pd.read_csv(train_filepath)
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    print(X[:5,:])
    print(y[:5])
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y_norm = (y - np.mean(y)) / np.std(y)
    return X_norm, y_norm

def plot_data(X, y):
    plt.scatter(X[:, 5], y, color='blue')
    plt.xlabel("Normalized Feature")
    plt.ylabel("Target Variable")
    plt.title("Feature vs Target")
    plt.show()

def compute_cost(y, y_pred):
    return np.mean((y_pred - y) ** 2)

def gradient_descent(X, y, alpha, iters):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    cost_history = []

    for i in range(iters):
        y_pred = np.dot(X, w) + b
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        w -= alpha * dw
        b -= alpha * db
        loss = compute_cost(y, y_pred)
        cost_history.append(loss)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")

    return w, b, cost_history

def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.grid(True)
    plt.show()

def evaluate_metrics(y, y_pred):
    mse = np.mean((y_pred - y) ** 2)
    mae = np.mean(np.abs(y_pred - y))
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return mse, mae, r_squared

train_filepath = "/content/drive/MyDrive/Machine Learning Library /Datasets/linear_regression_train.csv"
X_train, y_train = load_and_normalize_dataset(train_filepath)
print(X_train.shape)
print(y_train.shape)
print(X_train[:5,:])
print(y_train[:5])
plot_data(X_train, y_train)
alpha = 0.001
iters = 5000
w, b, cost_history = gradient_descent(X_train, y_train, alpha, iters)
plot_cost_history(cost_history)
y_train_pred = np.dot(X_train, w) + b
train_mse, train_mae, train_r_squared = evaluate_metrics(y_train, y_train_pred)
print("Training Metrics:")
print("Mean Squared Error: ", train_mse)
print("Mean Absolute Error: ",train_mae)
print("R-squared: ", train_r_squared)
