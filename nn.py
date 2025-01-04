import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1 - s)

def forward_propagation(X, weights, biases, activation_func=sigmoid):
    activations=[X]
    for i in range(len(weights)):
        net=np.dot(activations[-1], weights[i]) + biases[i]
        activations.append(activation_func(net))
    return activations

def backward_propagation(X, y, weights, activations, activation_derivative=sigmoid_derivative):
    m=X.shape[0]
    delta=activations[-1] - y.reshape(-1, 1)
    dW=[]
    db=[]
    for i in range(len(weights) - 1, -1, -1):
        dW.insert(0, np.dot(activations[i].T, delta) / m)
        db.insert(0, np.sum(delta, axis=0, keepdims=True) / m)
        if i > 0:
            delta = np.dot(delta, weights[i].T) * activation_derivative(activations[i])
    return dW, db

def update_parameters(weights, biases, dW, db, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate*dW[i]
        biases[i] -= learning_rate*db[i]
    return weights, biases

def compute_loss(y_true, y_pred):
    return np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)

def fit(X, y, layers, iters, learning_rate, activation_func=sigmoid, activation_derivative=sigmoid_derivative):
    layer_sizes = [X.shape[1]] + layers
    weights=[]
    biases=[]
    for i in range(1, len(layer_sizes)):
        weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]))
        biases.append(np.zeros((1, layer_sizes[i])))
    history = []

    for i in range(iters):
        activations=forward_propagation(X, weights, biases, activation_func)
        dW, db=backward_propagation(X, y, weights, activations, activation_derivative)
        weights, biases = update_parameters(weights, biases, dW, db, learning_rate)
        loss=compute_loss(y, activations[-1])
        print(f"loss: {loss}")
        history.append(loss)
    return weights, biases, history

def predict(X, weights, biases, activation_func=sigmoid):
    activations=forward_propagation(X, weights, biases, activation_func)
    predictions=activations[-1]
    return (predictions > 0.5).astype(int)

def evaluate_metrics(y_true, y_pred):
    tp=np.sum((y_true == 1)&(y_pred == 1))
    fp=np.sum((y_true == 0)&(y_pred == 1))  
    fn=np.sum((y_true == 1)&(y_pred == 0)) 
    if (tp + fp)>0:
        precision=tp / (tp + fp)
    else:
        precision=0
    if (tp + fn)>0:
        recall=tp/(tp + fn)
    else:
      recall=0
    if (precision+recall)>0:
        f1_score=2*(precision * recall)/(precision + recall)
    else:
        f1_score=0
    return precision, recall, f1_score

df=pd.read_csv("/content/drive/MyDrive/Machine Learning Library /Datasets/nn_train.csv")
X=df.iloc[:,1:-2].values
y_train_binary=df.iloc[:,-2].values
y_train_class=df.iloc[:,-1].values
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
layers = [16, 8, 1]
iters=100
learning_rate=0.01
weights, biases, history = fit(X_norm, y_train_binary, layers, iters, learning_rate)
y_pred = predict(X_norm, weights, biases)
precision, recall, f1_score = evaluate_metrics(y_train_binary, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
plt.plot(history)
plt.title("Loss during training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
