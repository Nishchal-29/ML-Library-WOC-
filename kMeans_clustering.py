import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def find_closest_centroids(X, centroids):
    K=centroids.shape[0]
    m=X.shape[0]
    idx=np.zeros(X.shape[0],dtype=int)
    for i in range(m):
        distance=[]
        for j in range(K):
            distance.append(np.linalg.norm(X[i]-centroids[j]))
        idx[i]=np.argmin(distance)
    return idx

def compute_centroids(X, idx, K):
    m, n=X.shape
    centroids=np.zeros((K,n))
    for k in range(K):
        points=X[idx==k]
        centroids[k]=np.mean(points,axis=0)
    return centroids

def run_Kmeans(X, initial_centroids, iters):
    m, n=X.shape
    K=initial_centroids.shape[0]
    centroids=initial_centroids
    idx=np.zeros(m,dtype=int)
    cost_history=[]
    for i in range(iters):
        idx=find_closest_centroids(X, centroids)
        centroids=compute_centroids(X, idx, K)
        print("K-Means iteration %d/%d" % (i, iters-1))
        cost=compute_cost(X, idx, centroids)
        cost_history.append(cost)
        if i % 2 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return centroids, idx, cost_history

def compute_cost(X, idx, centroids):
    K=centroids.shape[0]
    m=X.shape[0]
    cost=0
    for k in range(K):
        points=X[idx==k]
        cost+=np.sum(np.linalg.norm(points-centroids[k],axis=1)**2)
    return cost

def Kmeans_init_centroids(X, K):
    np.random.seed(31)
    m=X.shape[0]
    randidx=np.random.permutation(m)
    centroids=X[randidx[:K]]
    return centroids

def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.show()

def plot_clusters(X, idx, centroids):
    plt.figure(figsize=(10, 8))
    for k in range(centroids.shape[0]):
        cluster_points = X[idx == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {k+1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()

df=pd.read_csv("/content/drive/MyDrive/Machine Learning Library /Datasets/unsupervised_data.csv")
X_train=df.iloc[:,1:].values
print(X_train.shape)
print(X_train[:5,:])
X_train_norm=(X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)
print(X_train_norm[:5,:])
K=7
iters=10
initial_centroids=Kmeans_init_centroids(X_train_norm, K)
final_centroids, idx, cost_history=run_Kmeans(X_train_norm, initial_centroids, iters)
plot_cost_history(cost_history)
plot_clusters(X_train_norm, idx, final_centroids)
