import numpy as np
import matplotlib.pyplot as plt
import pandas as ps

def euclidean_distance(a,b):
    sum=np.sum((a-b)**2)
    distance=np.sqrt(sum)
    return distance

def knn(X_train, y_train, x, k):
    distances=[euclidean_distance(x,x_train) for x_train in X_train]
    indices=np.argsort(distances)[:k]
    labels=[y_train[i] for i in indices]
    return labels

def predict(X_train, y_train, X, k):
    results=[]
    for x in X:
        k_nearest_labels=knn(X_train, y_train, x, k)
        pred=np.argmax(np.bincount(k_nearest_labels))
        results.append(pred)
    return np.array(results)

np.random.seed(42)
n_samples = 20
n_features = 5
k=3
features = np.random.uniform(0, 10, size=(n_samples, n_features))
labels = np.random.choice([0, 1], size=n_samples)
columns = [f"Feature_{i+1}" for i in range(n_features)] + ["Label"]
dataset = pd.DataFrame(np.column_stack((features, labels)), columns=columns)

X_train=dataset.iloc[:,:-1].values
y_train=dataset.iloc[:,-1].values
X = np.array([[2.420553,6.721355,7.616196,2.376375,7.282163], [0.368869,6.095643,5.026790,0.514788,2.786465], [9.624473,2.517823,4.972485,3.008783,2.848405], [9.082659,2.395619,1.448949,4.894528,9.856505]])
predictions = predict(X_train, y_train, X, k)
print(predictions)
