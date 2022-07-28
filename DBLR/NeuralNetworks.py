from torch import nn

# ----------------------------------------------------------------------- #
#                Learn mean and variance diagonal for the                 #
#               variational approximation of the posterior.               #
# ----------------------------------------------------------------------- #


class NeuralNetworks(nn.Module):

    def __init__(self, input_dim, output_dim, hdim=0, train_mean=False, train_cov=False, train_reconstruction=False):

        super().__init__()

        self.train_mean = train_mean
        self.train_cov = train_cov
        self.train_reconstruction = train_reconstruction

        if self.train_reconstruction:
            # NN Layers
            self.hidden_reconstruction = nn.Linear(input_dim, output_dim)
        else:
            # NN Layers
            self.hidden = nn.Linear(input_dim, hdim)
            self.output = nn.Linear(hdim, output_dim)
            self.batchnorm = nn.BatchNorm1d(hdim)

        # Activation
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        if self.train_mean:
            # Mean
            out_mean = self.hidden(x)
            out_mean = self.batchnorm(out_mean)
            out_mean = self.relu(out_mean)
            self.mean = self.output(out_mean)

        if self.train_cov:
            # Diagonal covariance
            out_var = self.hidden(x)
            out_var = self.batchnorm(out_var)
            out_var = self.tanh(out_var)
            self.logvar = self.output(out_var)  # output is log(var)

        if self.train_reconstruction:
            out = self.hidden_reconstruction(x)
            self.theta = self.sigmoid(out)
