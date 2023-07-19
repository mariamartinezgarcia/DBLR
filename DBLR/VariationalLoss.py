import torch
import math
from torch import nn
from .NeuralNetworks import NeuralNetworks


class VariationalLoss(nn.Module):
    def __init__(self, D, hdim_mean, hdim_var, K, input_type, cuda_device):
        super(VariationalLoss, self).__init__()

        self.device = torch.device(
            "cuda:" + str(cuda_device) if torch.cuda.is_available() else "cpu"
        )

        # Dimensions
        self.d = D
        self.k = K

        # 1. Infer a low-dimensional representation of the observed data [q(Z|X)]
        # NN for the mean
        self.nn_mean_z = NeuralNetworks(self.d, self.k, hdim=hdim_mean, train_mean=True)
        # NN for the covariance
        self.nn_cov_z = NeuralNetworks(self.d, self.k, hdim=hdim_mean, train_cov=True)

        self.nn_mean_z.to(self.device)
        self.nn_cov_z.to(self.device)

        if input_type == "binary":
            # NNs for the reconstruction (binary input)
            self.nn_reconstruction_0 = NeuralNetworks(
                self.k, self.d, train_reconstruction=True
            )
            self.nn_reconstruction_1 = NeuralNetworks(
                self.k, self.d, train_reconstruction=True
            )

            self.nn_reconstruction_0.to(self.device)
            self.nn_reconstruction_1.to(self.device)

        if input_type == "real":
            # NNs for the reconstruction (real input)
            self.nn_reconstruction_0 = NeuralNetworks(
                self.k, self.d, hdim=hdim_mean, train_mean=True
            )
            self.nn_reconstruction_1 = NeuralNetworks(
                self.k, self.d, hdim=hdim_mean, train_mean=True
            )

            self.nn_reconstruction_0.to(self.device)
            self.nn_reconstruction_1.to(self.device)

        # 2. Infer the weights for classification [q(w|Z,y)]
        # NNs for the mean
        self.nn_mean_0 = NeuralNetworks(
            self.k, self.k + 1, hdim=hdim_mean, train_mean=True
        )
        self.nn_mean_1 = NeuralNetworks(
            self.k, self.k + 1, hdim=hdim_mean, train_mean=True
        )
        # NNs for the covariance
        self.nn_cov_0 = NeuralNetworks(
            self.k, self.k + 1, hdim=hdim_var, train_cov=True
        )
        self.nn_cov_1 = NeuralNetworks(
            self.k, self.k + 1, hdim=hdim_var, train_cov=True
        )

        self.nn_mean_0.to(self.device)
        self.nn_mean_1.to(self.device)
        self.nn_cov_0.to(self.device)
        self.nn_cov_1.to(self.device)

        # Loss
        self.bce = nn.BCELoss(reduction="none")
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    def compute_mean_cov_w(self, z, y, mask):
        """
        Compute mean and covariance for the product of multivariate Gaussians p(w|Z,y).

        Args:
            z (tensor) - tensor of shape (N,K), being N the number of observations
                and K the embedding dimensionality, containing a sample from q(Z|X).
            y (tensor) - tensor of shape (N,1), being N the number of observations,
                containing the target variable.
            mask (tensor) - tensor of shape (N,1), being N the number of observations, indicating
                the values observed and the values missing in the target variable.
        """

        # Select the observed samples (mask == 1)
        y_observed = y[mask == 1]
        z_observed = z[mask == 1, :]

        # Separate in batches according to the label value
        z0 = z_observed[torch.where(y_observed == 0)[0], :]
        z1 = z_observed[torch.where(y_observed == 1)[0], :]

        # COVARIANCE MATRIX #
        self.nn_cov_0.forward(z0)

        var0_batch = torch.exp(self.nn_cov_0.logvar)
        inv_var0 = 1 / var0_batch  # diagonal matrix
        var0 = 1 / torch.sum(inv_var0, axis=0)

        self.nn_cov_1.forward(z1)

        var1_batch = torch.exp(self.nn_cov_1.logvar)
        inv_var1 = 1 / var1_batch  # diagonal matrix
        var1 = 1 / torch.sum(inv_var1, axis=0)

        # Compute final parameters
        self.var_w = 1 / ((1 / var0) + (1 / var1))
        self.cov_w = torch.diag(self.var_w)

        # MEAN #
        self.nn_mean_0.forward(z0)
        mean0 = var0 * torch.sum(inv_var0 * self.nn_mean_0.mean, axis=0)

        self.nn_mean_1.forward(z1)
        mean1 = var1 * torch.sum(inv_var1 * self.nn_mean_1.mean, axis=0)

        # Compute final parameters
        self.mean_w = self.var_w * ((1 / var0) * mean0 + (1 / var1) * mean1)

    # --- SAMPLING --- #

    def sample_from_q_z(self):
        """
        Draw a sample from q(Z|X).
        """

        # Sampling from q(Z|X).
        # First, from N(0,I)
        # Then, scale by std vector and sum the mean

        # Shape [N, K]

        N = self.nn_mean_z.mean.shape[0]

        noise = torch.randn(N, self.k).to(self.device)
        self.sample_z = self.nn_mean_z.mean + noise * torch.sqrt(
            torch.exp(self.nn_cov_z.logvar)
        )

    def sample_from_q_w(self):
        """
        Draw a sample from q(w|Z,y).
        """
        # Sampling from q(w|Z,y).
        # First, sample from N(0,I)
        # Then, scale by std vector and sum the mean

        # Shape [K+1, 1] (w_0, ... , w_k)
        noise = torch.randn(self.k + 1).to(self.device)
        self.sample_w = (self.mean_w + noise * (torch.sqrt(self.var_w))).view(-1, 1)

    def sample_from_y(self, N):
        """
        Draw a sample from p(y|Z,w).

        Args:
            N (int) - number of input observations.
        """

        # Sampling from p(y|z,w)
        # First, sample from a Uniform distribution U(0,1)
        # Then, apply the reparameterization trick to sample from p(y|Z,X) in a differentiable manner
        # (Gumbel-Softmax trick).

        sigmoid = nn.Sigmoid()

        ones = torch.ones(N, 1).to(self.device)
        z = torch.cat((ones, self.nn_mean_z.mean), 1)  # [N, K+1]

        noise = torch.rand(N).to(self.device)
        theta = sigmoid(z @ self.mean_w)

        output = (
            torch.log(theta + 1e-20)
            - torch.log(1 - theta + 1e-20)
            + torch.log(noise + 1e-20)
            - torch.log(1 - noise + 1e-20)
        )
        output = output / 2  # temperature term

        self.sample_y = sigmoid(output)

    # --- ELBO --- #

    def bce_y(self, y, mask_y, mc=1):
        """
        Compute the E_q(Z,w)[log p(y|Z,w)] term.

        Args:
            y (tensor) - tensor of shape (N,1), being N the number of observations,
                containing the target variable.
            mask_y (tensor) - tensor of shape (N,1), being N the number of observations, indicating
                the values observed and the values missing in the target variable.
            mc (int, optional) - number of samples used for Monte Carlo
               approximation.
        Returns:
            bce_term (float) - E_q(Z,w)[log p(y|Z,w)] term.
        """

        # Select the observed samples (mask == 1)
        mask_y.to(self.device)
        y_observed = y[mask_y == 1]

        N = y.shape[0]
        ones = torch.ones(N, 1).to(self.device)

        bce_term = 0

        for i in range(mc):
            # Get the samplesp
            self.sample_from_q_z()  # [N, K]
            self.sample_from_q_w()  # [K+1, 1]

            # Add a column of ones to Z in order to take into account the intercept
            z = torch.cat((ones, self.sample_z), 1)  # [N, K+1]

            # Select the observed samples
            z_observed = z[mask_y == 1, :]

            entropy = -self.bce_logits(
                z_observed @ self.sample_w, y_observed
            )  # minus bce to cancel the negative sign of the loss
            # entropy = -self.bce_logits(z @ self.sample_w, y)
            entropy = torch.sum(entropy, axis=0)

            # Loss term
            bce_term += entropy

        return bce_term / torch.tensor(mc).to(self.device)

    def reconstruction_x(self, x, y, mask_x, mask_y, mc=1):
        """
        Compute the E_q(Z)[log p(X|Z,y)] term.

        Args:
            x (tensor) - tensor of shape (N,D), being N the number of samples
                and D the number of features, containing the input data.
            y (tensor) - tensor of shape (N,1), being N the number of observations,
                containing the target variable.
            mask_x (tensor) - tensor of shape (N,D), being N the number of observations
                and D the number of features, indicating the values observed and the values missing.
            mask_y (tensor) - tensor of shape (N,1), being N the number of observations, indicating
                the values observed and the values missing in the target variable.
            mc (int, optional) - number of samples used for Monte Carlo
               approximation.
        Returns:
            reconstruction_term (float) - E_q(Z)[log p(X|Z,y)] term.
        """

        elbo_term = 0

        mask_x.to(self.device)
        mask_y.to(self.device)

        x0 = x[torch.where(y == 0)[0], :]
        mask_x0 = mask_x[torch.where(y == 0)[0], :].to(self.device)

        x1 = x[torch.where(y == 1)[0], :]
        mask_x1 = mask_x[torch.where(y == 1)[0], :].to(self.device)

        # Select the samples with mask != 1 -> true missing y dropped for regularisation
        x_soft = x[(mask_y != 1), :]
        mask_xsoft = mask_x[(mask_y != 1), :].to(self.device)

        y_soft = y[mask_y != 1]

        for i in range(mc):
            # Get the sample
            self.sample_from_q_z()  # [N, K]

            z0 = self.sample_z[torch.where(y == 0)[0], :]
            z1 = self.sample_z[torch.where(y == 1)[0], :]
            z_soft = self.sample_z[(mask_y != 1), :]

            if self.input_type == "binary":
                # Reconstruction
                self.nn_reconstruction_0.forward(z0)
                bce_x_0 = torch.sum(
                    mask_x0 * (-self.bce(self.nn_reconstruction_0.theta, x0))
                )

                self.nn_reconstruction_1.forward(z1)
                bce_x_1 = torch.sum(
                    mask_x1 * (-self.bce(self.nn_reconstruction_1.theta, x1))
                )

                if z_soft.shape[0] != 0:
                    # Semisupervised
                    self.nn_reconstruction_0.forward(z_soft)
                    self.nn_reconstruction_1.forward(z_soft)
                    bce_x_soft_1 = torch.sum(
                        mask_xsoft
                        * (-self.bce(self.nn_reconstruction_1.theta, x_soft)),
                        1,
                    ).view(-1, 1)
                    bce_x_soft_0 = torch.sum(
                        mask_xsoft
                        * (-self.bce(self.nn_reconstruction_0.theta, x_soft)),
                        1,
                    ).view(-1, 1)

                    bce_x_soft = torch.sum(
                        y_soft * bce_x_soft_1 + (1 - y_soft) * bce_x_soft_0
                    )

                    elbo_term += bce_x_0 + bce_x_1 + bce_x_soft

                else:
                    # Fully supervised
                    elbo_term += bce_x_0 + bce_x_1

            if self.input_type == "real":
                # Reconstruction
                log_det_0 = torch.sum(mask_x0, dim=1) * torch.log(
                    torch.tensor(2 * math.pi).to(self.device)
                )
                log_det_1 = torch.sum(mask_x1, dim=1) * torch.log(
                    torch.tensor(2 * math.pi).to(self.device)
                )
                log_det_soft = torch.sum(mask_xsoft, dim=1) * torch.log(
                    torch.tensor(2 * math.pi).to(self.device)
                )

                self.nn_reconstruction_0.forward(z0)
                term_0 = torch.sum(
                    log_det_0
                    + torch.diag(
                        (x0 - mask_x0 * self.nn_reconstruction_0.mean)
                        @ (x0 - mask_x0 * self.nn_reconstruction_0.mean).T
                    )
                )

                self.nn_reconstruction_1.forward(z1)
                term_1 = torch.sum(
                    log_det_1
                    + torch.diag(
                        (x1 - mask_x1 * self.nn_reconstruction_1.mean)
                        @ (x1 - mask_x1 * self.nn_reconstruction_1.mean).T
                    )
                )

                if z_soft.shape[0] != 0:
                    # Semisupervised
                    self.nn_reconstruction_0.forward(z_soft)
                    self.nn_reconstruction_1.forward(z_soft)
                    term_soft_1 = torch.sum(
                        y_soft[:, 0]
                        * (
                            log_det_soft
                            + torch.diag(
                                (x_soft - mask_xsoft * self.nn_reconstruction_1.mean)
                                @ (
                                    x_soft - mask_xsoft * self.nn_reconstruction_1.mean
                                ).T
                            )
                        )
                    )
                    term_soft_0 = torch.sum(
                        (1 - y_soft[:, 0])
                        * (
                            log_det_soft
                            + torch.diag(
                                (x_soft - mask_xsoft * self.nn_reconstruction_0.mean)
                                @ (
                                    x_soft - mask_xsoft * self.nn_reconstruction_0.mean
                                ).T
                            )
                        )
                    )

                    elbo_term += term_1 + term_0 + term_soft_0 + term_soft_1
                else:
                    # Fully supervised
                    elbo_term += term_1 + term_0

        if self.input_type == "binary":
            result = elbo_term / torch.tensor(mc).to(self.device)

        if self.input_type == "real":
            result = -0.5 * (elbo_term / torch.tensor(mc).to(self.device))

        return result

    def kl_term_z(self):
        """
        Compute the Kullback-Leibler Divergence D_KL[q(Z|X)||p(Z)] term.

        Returns:
            kl (float) - D_KL[q(Z|X)||p(Z)] term.
        """

        var_z = torch.exp(self.nn_cov_z.logvar)

        kl = (-1 / 2) * torch.sum(
            var_z + self.nn_mean_z.mean**2 - 1 - self.nn_cov_z.logvar
        )

        return kl

    def kl_term_w(self):
        """
        Compute the Kullback-Leibler Divergence D_KL[q(w|Z,y)||p(w)] term.

        Returns:
            kl (float) - D_KL[q(w|Z,y)||p(w)] term.
        """

        kl = (-1 / 2) * torch.sum(
            self.var_w + self.mean_w**2 - 1 - torch.log(self.var_w)
        )

        return kl

    def compute_ELBO(self, x, y, mask_x, mask_y, mc=1):
        """
        Compute the ELBO loss.

        Args:
            x (tensor) - tensor of shape (N,D), being N the number of samples
                and D the number of features, containing the input data.
            y (tensor) - tensor of shape (N,1), being N the number of observations,
                containing the target variable.
            mask_x (tensor) - tensor of shape (N,D), being N the number of observations
                and D the number of features, indicating the values observed and the values missing.
            mask_y (tensor) - tensor of shape (N,1), being N the number of observations, indicating
                the values observed and the values missing in the target variable.
            mc (int, optional) - number of samples used for Monte Carlo
               approximation.
        """

        bce_y_term = self.bce_y(y, mask_y, mc)
        reconstruction_x_term = self.reconstruction_x(x, y, mask_x, mask_y, mc)
        kl_z = self.kl_term_z()
        kl_w = self.kl_term_w()

        self.ELBO_loss = -(bce_y_term + reconstruction_x_term + kl_w + kl_z)
        self.kl_z = kl_z
        self.kl_w = kl_w
        self.reconstruction_x_term = reconstruction_x_term
        self.bce_y_term = bce_y_term
