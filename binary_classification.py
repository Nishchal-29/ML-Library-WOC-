import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def load_dataset(train_filepath):
    df = pd.read_csv(train_filepath)
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    print(X[:5,:])
    print(y[:5])
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_norm, y

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def compute_cost(y, y_pred):
  return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def gradient_descent(X, y, alpha, iters):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    cost_history = []
    for i in range(iters):
        y_pred = sigmoid(np.dot(X, w) + b)
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        w -= alpha * dw
        b -= alpha * db
        loss = compute_cost(y, y_pred)
        cost_history.append(loss)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")
    return w, b

def evaluate_metrics(y, y_pred):
    y_pred_binary=(y_pred >=0.5).astype(int)
    tp=np.sum((y_pred_binary == 1)&(y == 1))
    fp=np.sum((y_pred_binary == 1)&(y == 0))
    fn=np.sum((y_pred_binary == 0)&(y == 1))
    if(tp + fp)>0:
      precision=tp / (tp + fp) 
    else:
      precision=0
    if(tp+fn)>0:
      recall=tp / (tp + fn)
    else:
      recall=0
    if(precision + recall)>0:
      f1_score = 2 * (precision * recall) / (precision + recall)
    else:
      f1_score=0
    return f1_score


train_filepath = "/content/drive/MyDrive/Machine Learning Library /Datasets/binary_classification_train.csv"
X_train, y_train = load_dataset(train_filepath)
alpha = 0.001
iters = 5000
w, b = gradient_descent(X_train, y_train, alpha, iters)
y_train_pred = sigmoid(np.dot(X_train, w) + b)
f1_score = evaluate_metrics(y_train, y_train_pred)
print(f"F1 score: {f1_score}")
