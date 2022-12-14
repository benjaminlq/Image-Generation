"""Models using Variational Auto Encoder
"""
from typing import Literal, Union

import torch
import torch.nn as nn

import config
from models.block import *
from models.vae import BaseVAE, ConvVAE


class BaseCVAE(BaseVAE):
    """
    Conditional Variational AutoEncoder. The network attempts to model the conditional probability
    distribution of the images in latent space. The prior probability conditioned on class C P(z|C)
    of the latent space is regularized to match a Standard Gaussian distribution. The training process
    involves learning the conditional probability (encoder) P(z|x, C) based on observed training data
    followed by learning the reconstruction function (decoder) to reconstruct the image P(x_hat|z,C)
    from latent vector z.
    """

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 256,
        hidden_size: int = config.HIDDEN_SIZE,
        emb_size: int = 128,
        num_classes: int = 10,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
        **kwargs,
    ):
        """Based Conditional VAE Model

        Args:
            input_size (int, optional): Input Image Dimension in format (C, H, W) Defaults to (1,28,28).
            common_size (int, optional): Output Size of shared architecture between feature extraction network and distribution estimation network. Defaults to 400.
            hidden_size (int, optional): Dimension of Latent representation. Defaults to 128.
            emb_size (int, optional): Size of label embedding vector. Defaults to 128.
            num_classes (int, optional): Number of training conditional classes. Defaults to 10.
            activation (Tanh or Sigmoid): Activation of the output. For BCELoss, activation function must be Sigmoid. Default to Sigmoid.
        """
        super(BaseCVAE, self).__init__(
            input_size=input_size,
            common_size=common_size,
            hidden_size=hidden_size,
            activation=activation,
            **kwargs,
        )

        self.emb_size = emb_size
        self.num_classes = num_classes
        self.input_embedding = nn.Embedding(num_classes, emb_size)

        self.mean_fc = nn.Linear(common_size + emb_size, hidden_size)
        self.var_fc = nn.Linear(common_size + emb_size, hidden_size)

        self.encoder = nn.Sequential(
            nn.Flatten(), MLPBlock(self.c * self.h * self.w, common_size)
        )
        self.decoder = nn.Sequential(
            MLPBlock(hidden_size + emb_size, common_size),
            nn.Linear(common_size, self.c * self.h * self.w),
            nn.Unflatten(1, (self.c, self.h, self.w)),
        )

    def encode(
        self, input_imgs: torch.tensor, input_classes: torch.tensor
    ) -> torch.tensor:
        """Encoder converts input images and input classes to Conditional probability distribution (Gaussian) of latent vectors P(z|x, y)

        Args:
            input_imgs (torch.tensor): Input Images
            input_classes (torch.tensor): Input Classes of input images

        Returns:
            torch.tensor: Mean and Diagonal Covariance Matrix of Gaussian describing the probability P(z|x, y)
        """
        # Input_imgs: (bs, c, h, w). Input_classes: (bs, )
        y = self.input_embedding(input_classes)  # (bs, cond_size)
        x = self.encoder(input_imgs)
        x_cond = torch.cat((x, y), dim=1)
        mu = self.mean_fc(x_cond)
        log_var = self.var_fc(x_cond)
        return mu, log_var

    def decode(self, z: torch.tensor, input_classes: torch.tensor) -> torch.tensor:
        """Reconstruction of latent vector z into images

        Args:
            z (torch.tensor): Latent vector z
            input_classes (torch.tensor): Input Classes for image reconstruction

        Returns:
            torch.tensor: Reconstructed Image based on input class. P(x_hat | z, y)
        """
        bs = z.size(0)
        y = self.input_embedding(input_classes)
        cond_z = torch.cat((z, y), dim=1)
        out = self.activation(self.decoder(cond_z))
        return out.view(bs, self.c, self.h, self.w)

    def forward(self, input_imgs: torch.tensor, input_classes: torch.tensor):
        """Forward Propagation. The steps involved in a forward pass:
        1. Given input image x and input class y, estimate the Gaussian distribution of latent representation
        P(z|x, y) ~ N(mu(z|x,y), std(z|x,y))
        2. Based P(z|x, y), sample a latent vector z
        3. Reconstruct image x_hat based on latent vector z and input classes y (Decode)

        Args:
            input_imgs (torch.tensor): Input Images
            input_classes (torch.tensor): Input Classes of input images

        Returns:
            tuple(torch.tensor, torch.tensor, torch.tensor): reconstructed images, latent mean, latent std
        """

        mu, log_var = self.encode(input_imgs, input_classes)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z, input_classes)
        return out, mu, log_var

    def generate(self, cond_class: Union[int, list] = 0):
        """Generate a sample or batch of samples randomly from prior P(z|C) ~ N(0,1)

        Args:
            cond_class (int|list): Class label or list of class labels to generate images. Defaults to 0.

        Returns:
            Tuple(torch.tensor, torch.tensor): (Generated Img Batch, Latent Vector Batch)
        """
        if isinstance(cond_class, int):
            cond_class = [cond_class]
        input_classes = torch.tensor(cond_class, dtype=torch.int32).to(config.DEVICE)
        zs = torch.randn((len(input_classes), self.hidden_size)).to(config.DEVICE)
        return (
            (self.decode(zs, input_classes).squeeze(0), zs.squeeze(0))
            if len(input_classes) == 1
            else (self.decode(zs, input_classes), zs)
        )

    def __str__(self):
        """Model Name"""
        return "BaseCVAE"


class DeepCVAE(BaseCVAE):
    """Deep Conditional Variational Auto Encoder with more linear FC layers."""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 256,
        hidden_size: int = config.HIDDEN_SIZE,
        emb_size: int = 128,
        num_classes: int = 10,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
    ):
        """Deep Conditional Variational Auto Encoder Module

        Args:
            input_size (int, optional): Input Image Dimension in format (C, H, W) Defaults to (1,28,28).
            common_size (int, optional): Output Size of shared architecture between feature extraction network and distribution estimation network. Defaults to 256.
            hidden_size (int, optional): Dimension of Latent representation. Defaults to 128.
            emb_size (int, optional): Size of label embedding vector. Defaults to 128.
            num_classes (int, optional): Number of training conditional classes. Defaults to 10.
            activation (Tanh or Sigmoid): Activation of the output. For BCELoss, activation function must be Sigmoid. Default to Sigmoid.
        """

        super(DeepCVAE, self).__init__(
            input_size, common_size, hidden_size, emb_size, num_classes, activation
        )

        self.mean_fc = nn.Sequential(
            MLPBlock(common_size + emb_size, 256), nn.Linear(256, hidden_size)
        )
        self.var_fc = nn.Sequential(
            MLPBlock(common_size + emb_size, 256), nn.Linear(256, hidden_size)
        )

        self.encoder = nn.Sequential(
            nn.Flatten(),
            MLPBlock(self.c * self.h * self.w, 784),
            MLPBlock(784, 392),
            MLPBlock(392, common_size),
        )

        self.decoder = nn.Sequential(
            MLPBlock(self.hidden_size + emb_size, 256),
            MLPBlock(256, 392),
            MLPBlock(392, 784),
            nn.Linear(784, self.c * self.h * self.w),
            nn.Unflatten(1, (self.c, self.h, self.w)),
        )

    def __str__(self):
        """Model Name"""
        return "DeepCVAE"


class ConvCVAE(BaseCVAE, ConvVAE):
    """Conditional VAE using convolution Encoder and Decoder for feature extraction and image reconstruction"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 256,
        hidden_size: int = config.HIDDEN_SIZE,
        emb_size: int = 128,
        num_classes: int = 10,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
        kernel_size: int = 3,
    ):
        """Convolutional Conditional Variational Auto Encoder Module

        Args:
            input_size (int, optional): Input Image Dimension in format (C, H, W) Defaults to (1,28,28).
            common_size (int, optional): Output Size of shared architecture between feature extraction network and distribution estimation network. Defaults to 256.
            hidden_size (int, optional): Dimension of Latent representation. Defaults to 128.
            cond_size (int, optional): Size of label embedding vector. Defaults to 128.
            num_classes (int, optional): Number of training conditional classes. Defaults to 10.
            activation (Tanh or Sigmoid): Activation of the output. For BCELoss, activation function must be Sigmoid. Default to Sigmoid.
        """

        super(ConvCVAE, self).__init__(
            input_size=input_size,
            common_size=common_size,
            hidden_size=hidden_size,
            emb_size=emb_size,
            num_classes=num_classes,
            activation=activation,
            kernel_size=kernel_size,
        )

        self.mean_fc = nn.Sequential(
            MLPBlock(common_size + emb_size, 256), nn.Linear(256, hidden_size)
        )
        self.var_fc = nn.Sequential(
            MLPBlock(common_size + emb_size, 256), nn.Linear(256, hidden_size)
        )

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
            MLPBlock(hidden_size + emb_size, common_size),
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
        """Model Name"""
        return "ConvCVAE"


if __name__ == "__main__":
    sample_imgs = torch.randn((5, 3, 28, 28)).to(config.DEVICE)
    c, h, w = sample_imgs.size(1), sample_imgs.size(2), sample_imgs.size(3)
    model = BaseCVAE(input_size=(c, h, w))
    model.to(config.DEVICE)
    model.eval()
    sample_labels = torch.randint(low=0, high=9, size=(5,)).to(config.DEVICE)
    out, mu, log_var = model(sample_imgs, sample_labels)
    generated_imgs, zs = model.generate(0)
    print(generated_imgs.size())
    print(zs.size())
    generated_imgs, zs = model.generate([3, 4])
    print(generated_imgs.size())
    print(zs.size())
