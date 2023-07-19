# Deep Bayesian Logistic Regression (DBLR) model

This code implements a package for the DBLR model, a general purpose classification and dimensionality reduction method. It is a semi-supervised Bayesian latent space model that infers a low-dimensional representation of high-dimensional data driven by the target label. During the inference process, the model also learns a global vector of weights that allows to make predictions given the low-dimensional representation of the observations. Given the probabilistic generative nature of the model, it can handle partially missing entries during training, including not annotated observations as censored samples. 

## Paper

The paper can be found at:
https://doi.org/10.1109/JBHI.2023.3279493
https://doi.org/10.1101/2022.12.18.520909

Proposed citation:

Martínez-García, M., & Olmos, P. M. (2023). Handling ill-conditioned omics data with deep probabilistic models. IEEE Journal of Biomedical and Health Informatics.

@article{martinez2023handling,
  title={Handling ill-conditioned omics data with deep probabilistic models},
  author={Mart{\'\i}nez-Garc{\'\i}a, Mar{\'\i}a and Olmos, Pablo M},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}

## Main contributions

- **Inference of the latent space driven by the label value.** The DBLR infers different low-dimensional latent distributions depending on the label value, forcing clusterisation in the latent space in an informative manner and thus capturing the underlying structure of the data.
- **Classification.** During the inference process, the model additionally learns a global vector of weights that allows to make predictions given the low-dimensional representation of the data.
- **Handling missing data.** As the DBLR is a probabilistic generative model, it can handle partially missing observations during the training process, including not annotated observations as censored samples.
- **Regularisation method to handle small datasets.** In order to handle small high-dimensional datasets, which usually entail overfitting problems, we introduced an additional regularisation mechanism following a drop-out-like strategy that relies in the generative and semi-supervised nature of the model.
- **Handling different data types.** We have defined and implemented different observation likelihood models that can be used to describe different data types. In particular, we show how to use the DBLR with binary and real-valued features.

## Calling from Python

**Instantiating**

```
from DBLR.DeepBLR import DeepBLR

dblr = DeepBLR(D, hdim_mean, hdim_var, K, input_type, percentage=0.2, cuda_device=1)
```
Args:
- D (int): dimensionality of the input data.
- hdim_mean (int): dimension of the hidden layer for the mean estimation neural networks (in the case of real data, it also defines the hidden dimension of the reconstruction neural network).
- hdim_var (int): dimension of the hidden layer for the covariance estimation neural networks.
- K (int): dimension of the latent space.
- input_type (string): input data type ('binary' or 'real').
- percentage (float, optional): percentage of labels set at missing (at random) for regularisation.
- cuda_device (int, optional): cuda device for GPU.

**Training**

```
dblr.train_epoch(x, y, batch_size=64, mc=1, verbose=True)
```
Args:
- x (tensor): tensor of shape (N,D), being N the number of observations and D the number of features, containing the input data.
- y (tensor): tensor of shape (N,1), being N the number of observations, containing the target variable.
- batch_size (int, optional): number of observations used at each SGD step.
- mc (int, optional): number of samples used for Monte Carlo approximation.
- verbose (boolean, optional): if True, print the ELBO loss value at the end of the epoch.

**Prediction**
```
dblr.predict(x, mc=200)
```
Args:
- x (tensor): tensor of shape (N,D), being N the number of samples and D the number of features, containing the input data.
- mc (int, optional) - number of samples used for Monte Carlo approximation.

Return:
- y (tensor): tensor of shape (N,1), being N the number of samples, containing the estimated probabilities.

*For more examples on how to use the model, check the example scripts provided.*

## Demos

- **Obtain a low-dimensional latent representation** -> `DBLR_toydata_example_embedding.py`
- **Supervised and semi-supervised classification** -> `DBLR_toydata_example_classification.py`
- **Missing value imputation** -> `DBLR_toydata_example_imputation.py`
