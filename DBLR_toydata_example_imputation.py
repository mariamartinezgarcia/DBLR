import torch
import numpy as np
import matplotlib.pyplot as plt
from DBLR.DeepBLR import DeepBLR
import seaborn as sns

torch.manual_seed(5)
np.random.seed(5)

K = 2       # Dimension of the low-dimensional space
D = 10000   # Number of input variables
N = 500     # Number of observations
input_type = 'real'

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
X = Z @ A + bias

# 3. Force missing samples in a random variable
p_missing = 0.5     # Percentage of missing observations for imputation
feature = np.random.choice(np.arange(D), 1)
index_nan = np.random.choice(np.arange(N), int(p_missing*N), replace=False)
x_missing = X.clone()
x_missing[index_nan, feature] = np.nan
print('Number of missings: ', torch.sum(x_missing.isnan()))

# --- Deep Bayesian Logistic Regression model --- #
dblr = DeepBLR(D, 100, 100, K, input_type, percentage=0, cuda_device=1)
for epoch in range(100):
    dblr.train_epoch(x_missing, Y, batch_size=N, mc=30, verbose=False)

# Obtain reconstruction from the model
x_imputed = dblr.impute(x_missing, Y)

plt.figure()
sns.distplot(X[index_nan, feature].cpu().data.numpy(), hist=True, kde=True, norm_hist=True, label='True')
sns.distplot(x_imputed[index_nan, feature].cpu().data.numpy(), hist=True, kde=True, norm_hist=True, label='DBLR')
plt.title('True vs imputed values')
plt.legend()
plt.show()