import torch
import numpy as np
from DBLR.DeepBLR import DeepBLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.manual_seed(5)

K = 2       # Dimension of the low-dimensional space
D = 10000   # Number of input variables
N = 500     # Number of observations
input_type = 'binary'

A = torch.randn((K, D))     # Random matrix to augment the dimensionality of the data
bias = torch.randn(D)       # Random bias vector to augment the dimensionality of the data

# 1. Sample Z.
mean1 = torch.randn(K) 
mean2 = torch.randn(K) 
mean3 = torch.rand(K) + 3
mean4 = torch.rand(K) + 3
G1 = torch.randn((int(N/4), K)) + mean1
G2 = torch.randn((int(N/4), K)) + mean2
G3 = torch.randn((int(N/4), K)) + mean3
G4 = torch.randn((int(N/4), K)) + mean4
Z = torch.cat((G1, G2, G3, G4), 0)

# Samples from G1 and G2 correspond to target 1, samples from G3 and G4 correspond to target 0
Y = torch.ones(N, 1)
Y[int(N/2):] = torch.tensor(0)

# 2. Generate X augmenting the dimensionality of the data and binarizing it.
probs_x = torch.sigmoid(Z @ A + bias)
X = torch.bernoulli(probs_x)

# Train / Test partition
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# --- Deep Bayesian Logistic Regression model --- #
# Fully Supervised
dblr = DeepBLR(D, 100, 100, K, input_type, percentage=0.2, cuda_device=0)
for epoch in range(50):
    dblr.train_epoch(X_train, Y_train, batch_size=N, mc=30, verbose=False)

Y_pred = dblr.predict(X_test)
roc_fs = roc_auc_score(Y_test, Y_pred)

print('ROC AUC DBLR = ', roc_fs)

# Semi Supervised
x_ss = torch.cat((X_train, X_test), 0)
y_ss = torch.cat((Y_train, Y_test), 0)
y_ss[Y_train.shape[0]:, :] = np.nan

dblr = DeepBLR(D, 100, 100, K, input_type, percentage=0.2, cuda_device=0)
for epoch in range(80):
    dblr.train_epoch(x_ss, y_ss, batch_size=N, mc=30, verbose=False)

Y_pred = dblr.predict(X_test)
roc_ss = roc_auc_score(Y_test, Y_pred)

print('ROC AUC DBLR SS = ', roc_ss)

