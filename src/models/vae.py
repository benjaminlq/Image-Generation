"""Models using Variational Auto Encoder
"""
from typing import Literal

import torch
import torch.nn as nn

import config
from models.block import *


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
        input_size: tuple = (1, 28, 28),
        common_size: int = 400,
        hidden_size: int = config.HIDDEN_SIZE,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
    ):
        """Based VAE Model

        Args:
            input_size (int, optional): Input Image Dimension in format (C, H, W) Defaults to (1,28,28).
            common_size (int, optional): Output Size of the shared architecture between network learning
            mean and variance of the estimated latent probability distribution. Defaults to 400.
            hidden_size (int, optional): Dimension of Latent representation. Defaults to 128.
            activation (Tanh or Sigmoid): Activation of the output. For BCELoss, activation function must be Sigmoid. Default to Sigmoid.
        """
        super(BaseVAE, self).__init__()

        self.c, self.h, self.w = input_size
        self.hidden_size = hidden_size
        self.common_size = common_size

        self.mean_fc = nn.Linear(common_size, hidden_size)
        self.var_fc = nn.Linear(common_size, hidden_size)

        self.encoder = nn.Sequential(
            nn.Flatten(), MLPBlock(self.c * self.h * self.w, common_size)
        )
        self.decoder = nn.Sequential(
            MLPBlock(hidden_size, common_size),
            nn.Linear(common_size, self.c * self.h * self.w),
            nn.Unflatten(1, (self.c, self.h, self.w)),
        )

        self.activation = getattr(nn, activation)()

    def encode(self, inputs: torch.tensor) -> torch.tensor:
        """Encoder converts input images to probability distribution (Gaussian) of latent vectors P(z|x)

        Args:
            inputs (torch.tensor): Input images

        Returns:
            Tuple(torch.tensor, torch.tensor): Mean and Diagonal Covariance Matrix of Gaussian describing the probability P(z|x)
        """
        x = self.encoder(inputs)
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
        bs = z.size(0)
        out = self.activation(self.decoder(z))
        return out.view(bs, self.c, self.h, self.w)

    def forward(self, inputs: torch.tensor):
        """Forward Propagation. The steps involved in a forward pass:
        1. Given input image x, estimate the Gaussian distribution of latent representation
        z of x. P(z|x) ~ N(mu(x), std(x))
        2. Based P(z|x), sample a latent vector z
        3. Reconstruct image x_hat based on latent vector z (Decode)

        Args:
            inputs (torch.tensor): Input Images

        Returns:
            tuple(torch.tensor, torch.tensor, torch.tensor): Reconstructed Images, Mean of Latent Distribution, Variance of Latent Distribution
        """

        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var

    def generate(self, num_samples: int = 1):
        """Generate a sample or batch of samples randomly from prior P(z) ~ N(0,1)

        Args:
            num_samples (int, optional): Size of sample batch to be generated. Defaults to 1.

        Returns:
            Tuple(torch.tensor, torch.tensor): (Generated Img Batch, Latent Vector Batch)
        """
        z = torch.randn((num_samples, self.hidden_size)).to(config.DEVICE)
        return (
            (self.decode(z).squeeze(0), z.squeeze(0))
            if num_samples == 1
            else (self.decode(z), z)
        )

    def __str__(self):
        """Model Name"""
        return "BaseVAE"


class DeepVAE(BaseVAE):
    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 128,
        hidden_size: int = config.HIDDEN_SIZE,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
    ):
        """Deep VAE Model with more linear layers than BaseVAE model.

        Args:
            input_size (int, optional): Input Image Dimension in format (C, H, W) Defaults to (1,28,28).
            common_size (int, optional): Output Size of the shared architecture between network learning
            mean and variance of the estimated latent probability distribution. Defaults to 400.
            hidden_size (int, optional): Dimension of Latent representation. Defaults to 128.
            activation (Tanh or Sigmoid): Activation of the output. For BCELoss, activation function must be Sigmoid. Default to Sigmoid.
        """
        super(DeepVAE, self).__init__(input_size, common_size, hidden_size, activation)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            MLPBlock(self.c * self.h * self.w, 784),
            MLPBlock(784, 392),
            MLPBlock(392, 196),
            MLPBlock(196, self.common_size),
        )

        self.decoder = nn.Sequential(
            MLPBlock(self.hidden_size, 128),
            MLPBlock(128, 196),
            MLPBlock(196, 392),
            MLPBlock(392, 784),
            nn.Linear(784, self.c * self.h * self.w),
            nn.Unflatten(1, (self.c, self.h, self.w)),
        )

    def __str__(self):
        """Model Name"""
        return "DeepVAE"


class ConvVAE(BaseVAE):
    """VAE using convolution Encoder and Decoder"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 196,
        hidden_size: int = config.HIDDEN_SIZE,
        activation: Literal["Tanh", "Sigmoid"] = "Tanh",
        kernel_size: int = 3,
    ):
        """Deep Convolutional VAE Model. Use Convolutional units for feature extraction (Downsampling) and Transpose Convolution (Upsampling)
        to reconstruct the images.

        Args:
            input_size (int, optional): Input Image Dimension in format (C, H, W) Defaults to (1,28,28).
            common_size (int, optional): Output Size of the shared architecture between network learning
            mean and variance of the estimated latent probability distribution. Defaults to 400.
            hidden_size (int, optional): Dimension of Latent representation. Defaults to 128.
            activation (Tanh or Sigmoid): Activation of the output. For BCELoss, activation function must be Sigmoid. Default to Sigmoid.
        """
        super(ConvVAE, self).__init__(input_size, common_size, hidden_size, activation)

        assert kernel_size % 2 == 1, "Kernel Size must be Odd"
        assert (
            self.h % 4 == 0 and self.w % 4 == 0
        ), "Height and Width of Original Image must be divisible by 4."
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.final_height = self.h // 4
        self.final_width = self.w // 4

        ## Input = c, h, w
        self.encoder = nn.Sequential(
            ConvBlock(
                self.c, 16, kernel_size=self.kernel_size, padding=self.padding
            ),  # 16, h, w
            DownSample(
                16, 32, kernel_size=self.kernel_size, padding=self.padding
            ),  # 32, h/2, w/2
            DownSample(
                32, 64, kernel_size=self.kernel_size, padding=self.padding
            ),  # 64, h/4, w/4
            nn.Flatten(),  # 64 & (h/4) * (w/4)
            MLPBlock(64 * self.final_height * self.final_width, 784),
            MLPBlock(784, 392),
            MLPBlock(392, common_size),
        )  # Common Size

        ## Input = hidden_size
        self.decoder = nn.Sequential(
            MLPBlock(hidden_size, common_size),
            MLPBlock(common_size, 392),
            MLPBlock(392, 784),
            MLPBlock(784, 64 * self.final_height * self.final_width),
            nn.Unflatten(
                1, (64, self.final_height, self.final_width)
            ),  # 64 & (h/4) * (w/4)
            UpSample(64, 32, kernel_size=self.kernel_size, padding=self.padding),
            UpSample(32, 16, kernel_size=self.kernel_size, padding=self.padding),
            ConvBlock(16, self.c, kernel_size=self.kernel_size, padding=self.padding),
        )

    def __str__(self):
        return "ConvVAE"


if __name__ == "__main__":
    sample = torch.rand(32, 3, 28, 28).to(config.DEVICE)
    c, h, w = sample.size(1), sample.size(2), sample.size(3)
    vae_model = ConvVAE(input_size=(c, h, w))
    vae_model.to(config.DEVICE)
    vae_model.eval()
    out, mu, log_var = vae_model(sample)
    generated_imgs, zs = vae_model.generate()
    print(generated_imgs.size())
    print(zs.size())
    generated_imgs, zs = vae_model.generate(3)
    print(generated_imgs.size())
    print(zs.size())
