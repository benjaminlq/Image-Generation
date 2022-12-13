"""Models using Variational Auto Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVAE(nn.Module):
    """
    Variational AutoEncoder. The network attempts to model the probability distribution
    of the latent manifold of the images. The prior probability of the latent space is regularized
    by assuming that the P(z) follows a Standard Gaussian distribution. The training process involves
    learning the conditional probability (encoder) P(z|x) based on observed training data followed by learning
    the reconstruction function (decoder) to reconstruct the image from latent vector z.
    """

    def __init__(
        self,
        input_size: int = 784,
        common_size: int = 400,
        hidden_size: int = 128,
        alpha: float = 1.0,
    ):
        """Based VAE Model

        Args:
            input_size (int, optional): Input Size. Defaults to 784.
            common_size (int, optional): Output Size of the shared architecture between network learning
            mean and variance of the estimated latent probability distribution. Defaults to 400.
            hidden_size (int, optional): _description_. Defaults to 20.
        """
        super(BaseVAE, self).__init__()

        self.hidden_size = hidden_size
        self.alpha = alpha

        self.flatten = nn.Flatten()
        self.common_fc = nn.Linear(input_size, common_size)
        self.mean_fc = nn.Linear(common_size, hidden_size)
        self.var_fc = nn.Linear(common_size, hidden_size)
        self.decode_fc = nn.Linear(hidden_size, common_size)
        self.out_fc = nn.Linear(common_size, input_size)

    def encode(self, flattened_inputs: torch.tensor) -> torch.tensor:
        """Encoder converts input images to probability distribution (Gaussian) of latent vectors P(z|x)

        Args:
            flattened_inputs (torch.tensor): Flattend input images

        Returns:
            torch.tensor: Mean and Diagonal Covariance Matrix of Gaussian describing the probability P(z|x)
        """
        x = F.relu(self.common_fc(flattened_inputs))
        mu = self.mean_fc(x)
        log_var = self.var_fc(x)
        return mu, log_var

    def reparameterize(self, mu: torch.tensor, log_var: torch.tensor) -> torch.tensor:
        """Reparameterization trick to allow backpropagation

        Args:
            mu (torch.tensor): Mean of the Gaussian distribution
            log_var (torch.tensor): Diagonal Covariance of the Gaussian distribution

        Returns:
            torch.tensor: Sampled latent vector z given x
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation of estimated P(z|x)
        eps = torch.randn_like(std)  # Sample on z ~ N(0,1)
        z = mu + eps * std  # Reparameterize z to on mu & std
        return z

    def decode(self, z: torch.tensor) -> torch.tensor:
        """Reconstruction of latent vector z into images

        Args:
            z (torch.tensor): Latent vector z

        Returns:
            torch.tensor: Reconstructed Image
        """
        h = F.relu(self.decode_fc(z))
        out = self.out_fc(h)
        return torch.sigmoid(out)

    def forward(self, inputs: torch.tensor):
        """Forward Propagation. The steps involved in a forward pass:
        1. Given input image x, estimate the Gaussian distribution of latent representation
        z of x. P(z|x) ~ N(mu(x), std(x))
        2. Based P(z|x), sample a latent vector z
        3. Reconstruct image x_hat based on latent vector z (Decode)

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.flatten(inputs)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var

    def __str__(self):
        """Model Name"""
        return "SimpleVAE"


if __name__ == "__main__":
    sample = torch.rand(3, 28, 28)
    vae_model = BaseVAE()
    out, mu, log_var = vae_model(sample)
    print(out.size())
    print(mu.size())
    print(log_var.size())
