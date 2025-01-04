import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def entropy(y):
  entropy=0
  if len(y)!=0:
    p1=len(y[y==1])/len(y)
    if p1!=0 and p1!=1:
      entropy=-p1*np.log2(p1)-(1-p1)*np.log2(1-p1)
    else:
      entropy=0
  return entropy

def split(X, indices, feature):
  left_indices=[]
  right_indices=[]
  for i in indices:
    if X[i][feature]<=0.5:
      left_indices.append(i)
    else:
      right_indices.append(i)
  return left_indices, right_indices

def information_gain(X, y, indices, feature):
  left_indices, right_indices=split(X, indices, feature)
  X_node, y_node=X[indices], y[indices]
  X_left, y_left=X[left_indices], y[left_indices]
  X_right, y_right=X[right_indices], y[right_indices]
  entropy_node=entropy(y_node)
  entropy_left=entropy(y_left)
  entropy_right=entropy(y_right)
  w_left=len(X_left)/len(X_node)
  w_right=len(X_right)/len(X_node)
  ig=entropy_node-(w_left*entropy_left+w_right*entropy_right)
  return ig

def best_split(X, y, indices):
  n=X.shape[1]
  best_ig=-1
  best_feature=-1
  for feature in range(n):
    ig=information_gain(X, y, indices, feature)
    if ig>best_ig:
      best_ig=ig
      best_feature=feature
  return best_feature

def build_tree(X, y, indices, branch_name, max_depth, curr_depth):
  tree=[]
  if curr_depth == max_depth:
        formatting = " "*curr_depth + "-"*curr_depth
        print(formatting, "%s leaf node with indices" % branch_name, indices)
        return
  best_feature = best_split(X, y, indices)
  tree.append((curr_depth, branch_name, best_feature, indices))
  formatting = "-"*curr_depth
  print("%s Depth %d, %s: Split on feature: %d" % (formatting, curr_depth, branch_name, best_feature))
  left_indices, right_indices = split(X, indices, best_feature)
  build_tree(X, y, left_indices, "Left", max_depth, curr_depth+1)
  build_tree(X, y, right_indices, "Right", max_depth, curr_depth+1)

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
indices = np.arange(len(y_train))
max_depth = 3
build_tree(X_train, y_train, indices, "Root", max_depth, 0)
