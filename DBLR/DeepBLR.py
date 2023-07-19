from .VariationalLoss import VariationalLoss
from collections import Counter
from torch import optim
import numpy as np
import torch


class DeepBLR(VariationalLoss):
    def __init__(
        self, D, hdim_mean, hdim_var, K, input_type, percentage=0.5, cuda_device=0
    ):
        super().__init__(D, hdim_mean, hdim_var, K, input_type, cuda_device)

        self.optimizer_mean_0 = optim.Adam(self.nn_mean_0.parameters(), lr=1e-3)
        self.optimizer_mean_1 = optim.Adam(self.nn_mean_1.parameters(), lr=1e-3)
        self.optimizer_cov_0 = optim.Adam(self.nn_cov_0.parameters(), lr=1e-3)
        self.optimizer_cov_1 = optim.Adam(self.nn_cov_1.parameters(), lr=1e-3)
        self.optimizer_mean_z = optim.Adam(self.nn_mean_z.parameters(), lr=1e-3)
        self.optimizer_cov_z = optim.Adam(self.nn_cov_z.parameters(), lr=1e-3)
        self.optimizer_reconstruction_0 = optim.Adam(
            self.nn_reconstruction_0.parameters(), lr=1e-3
        )
        self.optimizer_reconstruction_1 = optim.Adam(
            self.nn_reconstruction_1.parameters(), lr=1e-3
        )

        self.percentage = percentage  # percentage of missing entries for regularisation
        self.input_type = input_type  # binary or real

        self.ELBO_loss_epoch = []

    def sgd_step(self, x, y, mask_x, mask_y, mc=1):
        """
        Compute a SGD step.

        Args:
            x (tensor) - tensor of shape (N,D), being N the number of observations
                and D the number of features, containing the input data.
            y (tensor) - tensor of shape (N,1), being N the number of observations,
                containing the target variable.
            mask_x (tensor) - tensor of shape (N,D), being N the number of observations
                and D the number of features, indicating the values observed and the values missing.
            mask_y (tensor) - tensor of shape (N,1), being N the number of observations, indicating
                the values observed and the values missing.
            mc (int, optional) - number of samples used for Monte Carlo
               approximation.
        """

        # mask_y: array indicating the SAMPLES with the label at missing
        # mask_x: array indicating the SAMPLES with inputs at missing
        # mc: number of Monte Carlo samples
        # verbose: if True, print the ELBO loss

        self.optimizer_mean_0.zero_grad()
        self.optimizer_mean_1.zero_grad()
        self.optimizer_cov_0.zero_grad()
        self.optimizer_cov_1.zero_grad()
        self.optimizer_mean_z.zero_grad()
        self.optimizer_cov_z.zero_grad()
        self.optimizer_reconstruction_0.zero_grad()
        self.optimizer_reconstruction_1.zero_grad()

        x = x.to(self.device)
        y = y.to(self.device)

        # Compute mean and covariance matrix of the posterior distribution variational approximation q(Z|X)
        # We obtain one mean and variance vector for each observation (N vectors)
        self.nn_mean_z.forward(x)
        self.nn_cov_z.forward(x)

        # Compute mean and covariance matrix of the posterior distribution variational approximation q(w|Z,Y)
        # self.sample_from_q_z()
        self.compute_mean_cov_w(self.nn_mean_z.mean, y, mask=mask_y)

        # Sample from the generative model
        self.sample_from_y(x.shape[0])

        y_combined = y.clone()
        # Impute missing samples
        y_combined[mask_y == 0] = self.sample_y[mask_y == 0].clone().view(-1, 1)
        # Impute a percentage of the training samples
        y_combined[mask_y == 2] = self.sample_y[mask_y == 2].clone().view(-1, 1)

        # Compute the ELBO
        self.compute_ELBO(x, y_combined, mask_x, mask_y, mc)

        # Compute gradients
        self.ELBO_loss.backward()

        # Optimize parameters
        self.optimizer_mean_0.step()
        self.optimizer_cov_0.step()
        self.optimizer_mean_1.step()
        self.optimizer_cov_1.step()
        self.optimizer_mean_z.step()
        self.optimizer_cov_z.step()
        self.optimizer_reconstruction_0.step()
        self.optimizer_reconstruction_1.step()

    def train_epoch(self, x, y, batch_size=64, mc=1, verbose=True):
        """
        Train an epoch.

        Args:
            x (tensor) - tensor of shape (N,D), being N the number of observations
                and D the number of features, containing the input data.
            y (tensor) - tensor of shape (N,1), being N the number of observations,
                containing the target variable.
            batch_size (int, optional) - number of observations used at each SGD step.
            mc (int, optional) - number of samples used for Monte Carlo
               approximation.
            verbose (boolean, optional) - if True, print the ELBO loss value at the end of the epoch.
        """

        N = x.shape[0]

        # Aux mask to get balanced minibatches:
        #   0 -> observed sample with label 0
        #   1 -> observed sample with label 1
        #   2 -> missing sample

        mask_aux_minibatch = torch.ones(N)
        mask_aux_minibatch[y.isnan()[:, 0]] = 2
        # assign observed values to the observed samples
        mask_aux_minibatch[mask_aux_minibatch != 2] = y[mask_aux_minibatch != 2][:, 0]

        # Compute sample weights to obtain balanced minibatches
        count = Counter(mask_aux_minibatch.numpy().astype("int"))
        class_count = np.array([count[0], count[1], count[2]])
        weight = 1.0 / class_count
        samples_weight = np.array(
            [weight[t] for t in mask_aux_minibatch.numpy().astype("int")]
        )
        samples_weight = torch.from_numpy(samples_weight)
        # Generate DataLoader
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight)
        )
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler
        )

        self.mean_ELBO_loss_epoch = 0
        self.mean_bce_y_term_epoch = 0
        self.mean_reconstruction_x_term_epoch = 0
        self.mean_kl_z_term_epoch = 0
        self.mean_kl_w_term_epoch = 0

        count = 0

        # Loop through all the training samples
        for i, (data, target) in enumerate(dataloader):
            unique, counts = np.unique(target, return_counts=True)
            dicc = dict(zip(unique, counts))
            nans = len(target) - dicc[0] - dicc[1]

            # we need at leat 2 samples of each class
            if (0 in dicc.keys()) and (1 in dicc.keys()):
                if (dicc[0] <= 1) or (dicc[1] <= 1 or nans == 1):
                    continue
            else:
                continue

            # BALANCED MINIBATCH (1/3 observed label 0, 1/3 observed label 1, 1/3 missing)
            # Mask missing
            #   0 -> missing
            #   1 -> observed
            #   2 -> imputed

            mask_missing_minibatch = torch.ones(len(target))
            mask_missing_minibatch[target.isnan()[:, 0]] = 0

            # Force missing in a percentage of the observed samples for regularization
            n_imputed_samples = int(
                len(mask_missing_minibatch[mask_missing_minibatch == 1])
                * self.percentage
            )

            # Force missing in a balanced way
            index_observed_1 = np.where(target == 1)[0]
            index_observed_0 = np.where(target == 0)[0]

            if (len(index_observed_1) - int(n_imputed_samples / 2)) < 2:
                index_imputed_0 = np.sort(
                    np.random.choice(
                        index_observed_0, int(n_imputed_samples), replace=False
                    )
                )
                mask_missing_minibatch[index_imputed_0] = 2
            elif (len(index_observed_0) - int(n_imputed_samples / 2)) < 2:
                index_imputed_1 = np.sort(
                    np.random.choice(
                        index_observed_1, int(n_imputed_samples), replace=False
                    )
                )
                mask_missing_minibatch[index_imputed_1] = 2
            else:
                index_imputed_1 = np.sort(
                    np.random.choice(
                        index_observed_1, int(n_imputed_samples / 2), replace=False
                    )
                )
                index_imputed_0 = np.sort(
                    np.random.choice(
                        index_observed_0, int(n_imputed_samples / 2), replace=False
                    )
                )
                mask_missing_minibatch[index_imputed_1] = 2
                mask_missing_minibatch[index_imputed_0] = 2

            target_missing = target.clone()
            target_missing[mask_missing_minibatch == 2] = float("nan")

            # Set to zero the missing inputs and generate a mask
            mask_x_minibatch = torch.ones(data.shape)
            mask_x_minibatch[data.isnan()] = 0
            data[data.isnan()] = 0

            # SGD step
            self.sgd_step(
                data,
                target_missing,
                mask_x=mask_x_minibatch,
                mask_y=mask_missing_minibatch,
                mc=mc,
            )
            self.mean_ELBO_loss_epoch += -self.ELBO_loss.cpu().data.numpy()
            self.mean_bce_y_term_epoch += self.bce_y_term.cpu().data.numpy()
            self.mean_reconstruction_x_term_epoch += (
                self.reconstruction_x_term.cpu().data.numpy()
            )
            self.mean_kl_z_term_epoch += self.kl_z.cpu().data.numpy()
            self.mean_kl_w_term_epoch += self.kl_w.cpu().data.numpy()
            count += 1

        self.mean_ELBO_loss_epoch = self.mean_ELBO_loss_epoch / count
        self.mean_bce_y_term_epoch = self.mean_bce_y_term_epoch / count
        self.mean_reconstruction_x_term_epoch = (
            self.mean_reconstruction_x_term_epoch / count
        )
        self.mean_kl_z_term_epoch = self.mean_kl_z_term_epoch / count
        self.mean_kl_w_term_epoch = self.mean_kl_w_term_epoch / count

        if verbose:
            print("ELBO: ", -self.mean_ELBO_loss_epoch)

    def predict(self, x, mc=200):
        """
        Computes the probability of the label to take value 1 given the input data.

        Args:
            x (tensor) - tensor of shape (N,D), being N the number of samples
                and D the number of features, containing the input data.
            mc (int, optional) - number of samples used for Monte Carlo
                approximation.

        Return:
            y (tensor) - tensor of shape (N,1), being N the number of samples,
                containing the estimated probabilities.
        """

        N = x.shape[0]

        mask_x = torch.ones(x.shape)
        mask_x[x.isnan()] = 0
        x[x.isnan()] = 0

        predicted_y = torch.zeros(N, 1)
        predicted_y = predicted_y.to(self.device)

        x = x.to(self.device)

        # Obtain the projection into the latent space
        self.nn_mean_z.eval()
        self.nn_cov_z.eval()
        self.nn_mean_z.forward(x)
        self.nn_cov_z.forward(x)

        for i in range(mc):
            self.sample_from_q_z()
            self.sample_from_q_w()

            ones = torch.ones(N, 1)
            ones = ones.to(self.device)
            z = torch.cat((ones, self.sample_z), 1)
            z = z.to(self.device)
            predicted_y += torch.sigmoid(z @ self.sample_w)

        return predicted_y.data / mc

    def impute(self, x, y, mc=200):
        """
        Computes the probability of the label to take value 1 given the input data.

        Args:
            x (tensor) - tensor of shape (N,D), being N the number of samples
                and D the number of features, containing the input data.
            mc (int, optional) - number of samples used for Monte Carlo
                approximation.

        Return:
            X (tensor) - tensor of shape (N,D), being N the number of samples,
                containing the estimated values obtained sampling from the model.
        """

        self.nn_mean_z.eval()
        self.nn_cov_z.eval()
        self.nn_reconstruction_0.eval()
        self.nn_reconstruction_1.eval()

        x_zeros = x.clone()
        x_zeros[x_zeros.isnan()] = 0
        x_zeros.to(self.device)

        x_imputed = torch.zeros(x.shape).to(self.device)

        for i in range(mc):
            x_aux = torch.zeros(x.shape).to(self.device)
            # Sample from q(Z|X) to obtain the latent representation
            self.nn_mean_z.forward(x_zeros)
            self.nn_cov_z.forward(x_zeros)
            self.sample_from_q_z()

            # Sample from p(X|Z,y) to obtain the reconstruction
            z_0 = self.sample_z[torch.where(y == 0)[0], :]
            z_1 = self.sample_z[torch.where(y == 1)[0], :]
            self.nn_reconstruction_0.forward(z_0)
            self.nn_reconstruction_1.forward(z_1)

            if self.input_type == "real":
                N0 = self.nn_reconstruction_0.mean.shape[0]
                N1 = self.nn_reconstruction_1.mean.shape[0]
                noise0 = torch.randn(N0, x_zeros.shape[1]).to(self.device)
                noise1 = torch.randn(N1, x_zeros.shape[1]).to(self.device)
                x0 = self.nn_reconstruction_0.mean + noise0
                x1 = self.nn_reconstruction_1.mean + noise1
            if self.input_type == "binary":
                x0 = self.nn_reconstruction_0.theta
                x1 = self.nn_reconstruction_1.theta

            x_aux[torch.where(y == 0)[0], :] = x0
            x_aux[torch.where(y == 1)[0], :] = x1

            x_imputed += x_aux

        x_imputed = x_imputed / mc

        if self.input_type == "binary":
            x_imputed = torch.bernoulli(x_imputed)

        return x_imputed
