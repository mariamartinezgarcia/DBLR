import torch
import matplotlib.pyplot as plt
from DBLR.DeepBLR import DeepBLR

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

# Plot the original latent space
plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=Y)
plt.title('Original latent space')
plt.show()

# 2. Generate X augmenting the dimensionality of the data and binarizing it.
probs_x = torch.sigmoid(Z @ A + bias)
X = torch.bernoulli(probs_x)

# --- Deep Bayesian Logistic Regression model --- #
dblr = DeepBLR(D, 100, 100, K, input_type, percentage=0, cuda_device=0)
for epoch in range(50):
    dblr.train_epoch(X, Y, batch_size=N, mc=30, verbose=False)

# Obtain the latent representation of the data using the encoder networks
dblr.nn_mean_z.eval()
dblr.nn_cov_z.eval()
dblr.nn_mean_z.forward(X)
dblr.nn_cov_z.forward(X)
z_dblr = dblr.nn_mean_z.mean.cpu().data.numpy()

z_dblr = 0
mc = 300    # Number of samples for Monte Carlo approximation
for i in range(mc):
    dblr.sample_from_q_z()
    z_dblr += dblr.sample_z.cpu().data.numpy()
z_dblr = z_dblr/mc

# Plot the obtained low-dimensional representation
plt.figure()
plt.scatter(z_dblr[:, 0], z_dblr[:, 1], c=Y.numpy())
plt.title('Deep BLR')
plt.show()
