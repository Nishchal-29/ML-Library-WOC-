import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_normalize_train_dataset(train_filepath):
    df = pd.read_csv(train_filepath)
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    print(X[:5, :])
    print(y[:5])
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_norm, y

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_cost(y, y_pred, m):
    log_likelihood = -np.log(y_pred[range(m), y])
    return np.sum(log_likelihood) / m

def gradient_descent(X, y, alpha, iters, n_classes):
    m, n = X.shape
    w = np.zeros((n, n_classes))
    b = np.zeros(n_classes)
    cost_history = []
    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y] = 1

    for i in range(iters):
        logits = np.dot(X, w) + b
        y_pred = softmax(logits)
        dw = (1 / m) * np.dot(X.T, (y_pred - y_one_hot))
        db = (1 / m) * np.sum(y_pred - y_one_hot, axis=0)
        w -= alpha * dw
        b -= alpha * db
        cost = compute_cost(y, y_pred, m)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {cost:.4f}")
    return w, b, cost_history

def evaluate_metrics(X, y, w, b):
    logits=np.dot(X, w) + b
    y_pred=np.argmax(softmax(logits), axis=1)
    accuracy=np.mean(y_pred == y)
    n_classes=len(y)
    precision=[]
    recall=[]
    
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y == c))
        fp = np.sum((y_pred == c) & (y != c))
        fn = np.sum((y_pred != c) & (y == c))
        if(tp+fp)>0:
          class_precision=tp/(tp+fp)
        else:
          class_precision=0
        if(tp+fn)>0:
          class_recall=tp/(tp+fn)
        else:
          class_recall=0     
        precision.append(class_precision)
        recall.append(class_recall)
    f1_scores = []
    for i in range(n_classes):
        if precision[i] + recall[i] > 0:
            f1=2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1=0
        f1_scores.append(f1)
    class_weights = [np.sum(y == c) / len(y) for c in range(n_classes)]
    f1_score = np.sum([f1 * weight for f1, weight in zip(f1_scores, class_weights)])    
    return accuracy, f1_score

train_filepath = "/content/drive/MyDrive/Machine Learning Library /Datasets/multi_classification_train.csv"
X_train, y_train = load_and_normalize_train_dataset(train_filepath)

n_classes = len(np.unique(y_train))
alpha = 0.01
iters = 5000
w, b, cost_history = gradient_descent(X_train, y_train, alpha, iters, n_classes)
accuracy, f1_score = evaluate_metrics(X_train, y_train, w, b)
print(f"Training Accuracy: {accuracy}")
print(f"Training F1 Score: {f1_score}")
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History")
plt.grid()
plt.show()
