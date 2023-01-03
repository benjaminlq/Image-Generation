"""GAN Model
"""
from typing import Callable, Union

import torch
import torch.nn as nn
from torchvision import transforms

import config
from models.block import *


class Generator(nn.Module):
    """Generator Module."""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        hidden_size: int = 64,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ):
        """Generate Images by sampling from a Standard Gaussian prior distribution to and transform them
        through Generator NN. The goal of Generator is to produce generation distribution as close to real images distribution
        as possible by minimizing the Jenson-Shannon divergence between the two distribution. Optimization is done by a second neural
        network, the Discriminator, which can be regarded as the loss function to train the Generator. The Generator tries to trick
        the Discriminator into misclassifying generated images as real images, which ultimately minimizes the Jenson-Shannon divergence
        between the generated distribution and the real distribution. The training goal is to achieve a Nash equilibrium whereby at location,
        P_D_real = P_D_gen = 0.5, and the Generator stops at its global optimum loss where generated distribution matches real distribution.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28).
            hidden_size (int, optional): Hidden size of input noise latent vector to Generator. Defaults to 64.
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.2.
        """
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.img_shape = input_size
        self.num_classes = num_classes
        self.conditional = False

        self.linear_1 = MLPBlock(self.hidden_size, 128)
        self.linear_2 = MLPBlock(128, 256)
        self.linear_3 = MLPBlock(256, 512)
        self.linear_4 = MLPBlock(512, 1024)
        self.linear_out = nn.Linear(
            1024, self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.tensor) -> torch.tensor:
        """Forward Pass

        Args:
            z (torch.tensor): Noise Latent vectors to feed to Generator

        Returns:
            torch.tensor: Generated Images
        """
        x = self.dropout(self.linear_1(z))
        x = self.dropout(self.linear_2(x))
        x = self.dropout(self.linear_3(x))
        x = self.dropout(self.linear_4(x))
        x = self.linear_out(x)
        x = self.tanh(x)
        imgs = x.view(x.size(0), *self.img_shape)

        return imgs

    def generate(self, num_samples: int = 1):
        """Randomly Generate a batch of images from generator

        Args:
            num_samples (int, optional): Number of samples in random batch. Default to 1.

        Returns:
            Tuple(torch.tensor, torch.tensor): Generated Images, Input Latent Noise Vectors.
        """

        z = torch.randn((num_samples, self.hidden_size), device=config.DEVICE)
        imgs = (self(z) + 1) / 2
        return (imgs.squeeze(0), z.squeeze(0)) if num_samples == 1 else (imgs, z)

    def __str__(self):
        """Model Name"""
        return "GAN"


class Discriminator(nn.Module):
    """Discriminator Module"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        dropout_rate: float = 0.4,
        num_classes: int = 10,
    ):
        """Discriminator Module attempts to differentiate generated images from real images, allowing the Generator distribution
        to move closer to real image distribution. At optimum, Discriminator prediction is D(x) = p_real(x) / (p_real(x) + p_gen(x)).
        This is the loss function to use to minimize Jenson-Shannon divergence between generate distribution and real distribution.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28).
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.4.
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
        """
        super(Discriminator, self).__init__()
        self.img_shape = input_size
        self.num_classes = num_classes
        self.conditional = False

        self.linear_1 = nn.Linear(
            self.img_shape[0] * self.img_shape[1] * self.img_shape[2], 784
        )
        self.linear_2 = nn.Linear(784, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

        self.activation = nn.LeakyReLU(0.1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, imgs: torch.tensor) -> torch.tensor:
        """Forward Pass

        Args:
            imgs (torch.tensor): Generated or Real Images

        Returns:
            torch.tensor: Probabilities of images being real.
        """
        x = self.flatten(imgs)
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.dropout(self.activation(self.linear_2(x)))
        x = self.dropout(self.activation(self.linear_3(x)))
        probs = self.out(x)
        return probs

    def __str__(self):
        """Model Name"""
        return "Discriminator"


class CGenerator(Generator):
    """Conditional Generator Module"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        hidden_size: int = 64,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        emb_size: int = 32,
    ):
        """Conditional Generator Module. Concatenate class label to control which class the generated image belong to.
        The generator will model conditional probability P(z|C) when generating images conditioned on class labels.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28).
            hidden_size (int, optional): Hidden size of input noise latent vector to Generator. Defaults to 64.
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.2.
            emb_size (int, optional): Size of class embedding vector. Defaults to 32.
        """
        super(CGenerator, self).__init__(
            input_size, hidden_size, num_classes, dropout_rate
        )
        self.conditional = True
        self.emb_size = emb_size
        self.embedding = nn.Embedding(num_classes, emb_size)
        self.linear_1 = MLPBlock(hidden_size + emb_size, 128)

    def forward(
        self, z: torch.tensor, cond_class: Union[int, torch.tensor]
    ) -> torch.tensor:
        """Forward Pass

        Args:
            z (torch.tensor): Noise Latent vectors to feed to Generator
            cond_class (Union[int, list]): Class for conditional GAN. Model P(z|c)

        Returns:
            torch.tensor: Generated Images
        """
        if isinstance(cond_class, int):
            cond_class = torch.tensor([cond_class], dtype=torch.int32)
        input_embs = self.embedding(cond_class.to(config.DEVICE))
        z = torch.cat((z, input_embs), dim=1)

        x = self.dropout(self.linear_1(z))
        x = self.dropout(self.linear_2(x))
        x = self.dropout(self.linear_3(x))
        x = self.dropout(self.linear_4(x))
        x = self.linear_out(x)
        x = self.tanh(x)
        imgs = x.view(x.size(0), *self.img_shape)

        return imgs

    def generate(self, cond_class: Union[int, list] = 0):
        """Randomly Generate a batch of images from generator based on list of class inputs

        Args:
            cond_class (Union[int, list], optional): Class inputs for conditional GAN. Defaults to 0.

        Returns:
            Tuple(torch.tensor, torch.tensor): Generated Images, Input Latent Noise Vectors.
        """
        if isinstance(cond_class, int):
            cond_class = [cond_class]
        input_classes = torch.tensor(
            cond_class, dtype=torch.int32, device=config.DEVICE
        )
        zs = torch.randn((len(input_classes), self.hidden_size), device=config.DEVICE)
        imgs = (self(zs, input_classes) + 1) / 2
        return (
            (imgs.squeeze(0), zs.squeeze(0)) if len(input_classes) == 1 else (imgs, zs)
        )

    def __str__(self):
        """Model Name"""
        return "CGAN"


class CDiscriminator(Discriminator):
    """Conditional Discriminator Module"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        dropout_rate: float = 0.4,
        num_classes: int = 10,
        emb_size: int = 32,
    ):
        """Conditional Discriminator Module. Concatenate class label with image features during discrimination loss training. The discriminator will model conditional probability P(X=real|C) when classifying images conditioned on class labels.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28)
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.4
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
            emb_size (int, optional): Size of class embedding vector. Defaults to 32.
        """
        super(CDiscriminator, self).__init__(input_size, dropout_rate, num_classes)
        self.conditional = True
        self.embedding = nn.Embedding(num_classes, emb_size)
        self.linear_1 = nn.Linear(
            self.img_shape[0] * self.img_shape[1] * self.img_shape[2] + emb_size, 784
        )

    def forward(
        self, imgs: torch.tensor, cond_class: Union[int, torch.tensor]
    ) -> torch.tensor:
        """Forward Pass

        Args:
            imgs (torch.tensor): Real or Fake Images
            cond_class (Union[int, torch.tensor]): Class for input images.

        Returns:
            torch.tensor: Probabilities of images being real.
        """
        x = self.flatten(imgs)
        if isinstance(cond_class, int):
            cond_class = torch.tensor([cond_class, int], dtype=torch.int32)
        input_embs = self.embedding(cond_class.to(config.DEVICE))
        x = torch.cat((x, input_embs), dim=1)
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.dropout(self.activation(self.linear_2(x)))
        x = self.dropout(self.activation(self.linear_3(x)))
        probs = self.out(x)
        return probs

    def __str__(self):
        """Model Name"""
        return "CDiscriminator"


class ConvGenerator(Generator):
    """Deep Convolution Generator Module."""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        hidden_size: int = 64,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        #    emb_size: int = 32,
    ):
        """Deep Convolution Generator Module. Following DCGAN paper for Transpose Convolution for image generation.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28).
            hidden_size (int, optional): Hidden size of input noise latent vector to Generator. Defaults to 64.
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.2.
        """
        super(ConvGenerator, self).__init__(
            input_size, hidden_size, num_classes, dropout_rate
        )
        self.conditional = False
        self.resize = transforms.Resize(self.img_shape[1])
        self.conv_transpose_1 = ConvTranspose2DBlock(
            hidden_size, 256, 4, 1, 0, bias=True
        )
        self.conv_transpose_2 = ConvTranspose2DBlock(
            256, 128, 4, 2, 1, bias=True
        )  # 128, 8, 8
        self.conv_transpose_3 = ConvTranspose2DBlock(
            128, 64, 4, 2, 1, bias=True
        )  # 64, 16, 16
        self.out = nn.ConvTranspose2d(
            64, self.img_shape[0], 4, 2, 1
        )  # no_channels, 32, 32
        self.apply(self.init_weights)

    def forward(self, z: torch.tensor) -> torch.tensor:
        """Forward Pass

        Args:
            z (torch.tensor): Noise Latent vectors to feed to Generator

        Returns:
            torch.tensor: Generated Images
        """
        z = z.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_transpose_1(z)
        x = self.conv_transpose_2(x)
        x = self.conv_transpose_3(x)
        imgs = self.tanh(self.out(x))
        return imgs

    def generate(self, num_samples: int = 1):
        """Randomly Generate a batch of images from generator.

        Args:
            num_samples (int, optional): Number of samples in random batch. Default to 1.

        Returns:
            Tuple(torch.tensor, torch.tensor): Generated Images, Input Latent Noise Vectors.
        """
        z = torch.randn((num_samples, self.hidden_size), device=config.DEVICE)
        imgs = (self(z) + 1) / 2
        imgs = self.resize(imgs)
        return (imgs.squeeze(0), z.squeeze(0)) if num_samples == 1 else (imgs, z)

    @staticmethod
    def init_weights(m: Callable):
        """Initialize weights and biases

        Args:
            m (Callable): Module with parameters to be optimized
        """
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    def __str__(self):
        """Model Name"""
        return "ConvGAN"


class ConvDiscriminator(nn.Module):
    """Deep Convolutional Discriminator Module"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        num_classes: int = 10,
    ):
        """Deep Convolutional Discriminator Module. Use 2D Convolution for feature extraction.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 32, 32).
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
        """
        super(ConvDiscriminator, self).__init__()
        self.conditional = False
        self.img_shape = input_size
        self.num_classes = num_classes
        self.resize = transforms.Resize(32)
        self.convblock_1 = nn.Sequential(
            nn.Conv2d(self.img_shape[0], 32, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock_2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_out = nn.Conv2d(128, 1, 4, 1, 0)
        self.apply(self.init_weights)

    def forward(self, imgs: torch.tensor) -> torch.tensor:
        """Forward Pass

        Args:
            imgs (torch.tensor): Real and Generated images for discriminator classification task.

        Returns:
            torch.tensor: Probability of image being real.
        """
        # bs, c, h, w
        imgs = self.resize(imgs)  # (c, 32, 32)
        x = self.convblock_1(imgs)  # (64, 8, 8)
        x = self.convblock_2(x)  # (128, 4, 4)
        out = self.conv_out(x)  # (1, 1, 1)
        return out.view(imgs.size(0), -1)

    @staticmethod
    def init_weights(m):
        """Initialize weights and biases

        Args:
            m (Callable): Module with parameters to be optimized
        """
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    def __str__(self):
        """Model Name"""
        return "ConvDiscriminator"


class ConvCGenerator(CGenerator):
    """Deep Convolution Generator Module."""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        hidden_size: int = 64,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        emb_size: int = 32,
    ):
        """Deep Convolution Generator Module. Following DCGAN paper for Transpose Convolution for image generation.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28).
            hidden_size (int, optional): Hidden size of input noise latent vector to Generator. Defaults to 64.
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.2.
        """
        super(ConvCGenerator, self).__init__(
            input_size, hidden_size, num_classes, dropout_rate, emb_size
        )
        self.conditional = True
        self.resize = transforms.Resize(self.img_shape[1])
        self.embedding = nn.Embedding(num_classes, emb_size)
        self.conv_transpose_1 = ConvTranspose2DBlock(
            hidden_size, 256, 4, 1, 0, bias=True
        )
        self.conv_label = ConvTranspose2DBlock(emb_size, 128, 4, 1, 0, bias=True)
        self.conv_transpose_2 = ConvTranspose2DBlock(
            256 + 128, 128, 4, 2, 1, bias=True
        )  # 128, 8, 8
        self.conv_transpose_3 = ConvTranspose2DBlock(
            128, 64, 4, 2, 1, bias=True
        )  # 64, 16, 16
        self.out = nn.ConvTranspose2d(
            64, self.img_shape[0], 4, 2, 1
        )  # no_channels, 32, 32
        self.apply(self.init_weights)

    def forward(
        self, z: torch.tensor, cond_class: Union[int, torch.tensor]
    ) -> torch.tensor:
        """Forward Pass

        Args:
            z (torch.tensor): Noise Latent vectors to feed to Generator
            cond_class (Union[int, list]): Class for conditional GAN. Model P(z|c)

        Returns:
            torch.tensor: Generated Images
        """
        # z: bs, hidden_size
        if isinstance(cond_class, int):
            cond_class = torch.tensor([cond_class], dtype=torch.int32)

        input_embs = self.embedding(cond_class.to(config.DEVICE))  # bs, emb_size

        x1 = self.conv_transpose_1(z.unsqueeze(-1).unsqueeze(-1))
        x2 = self.conv_label(input_embs.unsqueeze(-1).unsqueeze(-1))
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_transpose_2(x)
        x = self.conv_transpose_3(x)
        imgs = self.tanh(self.out(x))
        return imgs

    def generate(self, cond_class: Union[int, list] = 0):
        """Randomly Generate a batch of images from generator based on list of class inputs

        Args:
            cond_class (Union[int, list], optional): Class inputs for conditional GAN. Defaults to 0.

        Returns:
            Tuple(torch.tensor, torch.tensor): Generated Images, Input Latent Noise Vectors.
        """
        if isinstance(cond_class, int):
            cond_class = [cond_class]
        input_classes = torch.tensor(
            cond_class, dtype=torch.int32, device=config.DEVICE
        )
        zs = torch.randn((len(input_classes), self.hidden_size), device=config.DEVICE)
        imgs = (self(zs, input_classes) + 1) / 2
        imgs = self.resize(imgs)
        return (
            (imgs.squeeze(0), zs.squeeze(0)) if len(input_classes) == 1 else (imgs, zs)
        )

    @staticmethod
    def init_weights(m):
        """Initialize weights and biases

        Args:
            m (Callable): Module with parameters to be optimized
        """
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    def __str__(self):
        """Model Name"""
        return "ConvCGAN"


class ConvCDiscriminator(nn.Module):
    """Conditional Discriminator Module"""

    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        num_classes: int = 10,
    ):
        """Conditional Discriminator Module. Concatenate class label with image features during discrimination loss training. The discriminator will model conditional probability P(X=real|C) when classifying images conditioned on class labels.

        Args:
            input_size (tuple, optional): Size of images to be generated. Defaults to (1, 28, 28)
            dropout_rate (float, optional): Dropout Rate to use for dropout units. Defaults to 0.4
            num_classes (int, optional): Number of conditional classes in each dataset. Defaults to 10.
            emb_size (int, optional): Size of class embedding vector. Defaults to 32.
        """
        super(ConvCDiscriminator, self).__init__()
        self.conditional = True
        self.in_channels = input_size[0]
        self.img_size = 32
        self.num_classes = num_classes
        self.resize = transforms.Resize(self.img_size)

        self.conv_1 = nn.Conv2d(self.in_channels, 32, 4, 2, 1)
        self.conv_label = nn.Conv2d(self.num_classes, 32, 4, 2, 1)
        self.conv_2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.conv_3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.conv_out = nn.Conv2d(256, 1, 4, 1, 0)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply(self.init_weights)

    def forward(
        self, imgs: torch.tensor, cond_class: Union[int, torch.tensor]
    ) -> torch.tensor:
        """Forward Pass

        Args:
            imgs (torch.tensor): Real or Fake Images
            cond_class (Union[int, torch.tensor]): Class for input images.

        Returns:
            torch.tensor: Probabilities of images being real.
        """
        # Imgs: bs, c, h, w
        if isinstance(cond_class, int):
            cond_class = torch.tensor(
                [cond_class, int], dtype=torch.int32, device=config.DEVICE
            )

        labels = torch.zeros(
            (len(cond_class), self.num_classes, self.img_size, self.img_size),
            device=config.DEVICE,
        )
        for idx, label in enumerate(cond_class):
            labels[idx, label.item(), :, :] = torch.ones(
                (self.img_size, self.img_size), device=config.DEVICE
            )

        x = self.leaky_relu(self.conv_1(self.resize(imgs)))
        x_label = self.leaky_relu(self.conv_label(labels))
        x = torch.cat((x, x_label), dim=1)
        x = self.leaky_relu(self.bn_2(self.conv_2(x)))
        x = self.leaky_relu(self.bn_3(self.conv_3(x)))
        probs = self.conv_out(x)
        return probs.view(imgs.size(0), -1)

    @staticmethod
    def init_weights(m):
        """Initialize weights and biases

        Args:
            m (Callable): Module with parameters to be optimized
        """
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    def __str__(self):
        """Model Name"""
        return "ConvCDiscriminator"


if __name__ == "__main__":
    generator = ConvCGenerator().to(config.DEVICE)
    sample_zs = torch.randn(size=(5, 64)).to(config.DEVICE)
    cond_class = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=config.DEVICE)
    output_imgs = generator(sample_zs, cond_class)
    print(str(generator), output_imgs.size())
    discriminator = ConvCDiscriminator(input_size=(1, 28, 28)).to(config.DEVICE)
    probs = discriminator(output_imgs, cond_class)
    print(str(discriminator), probs.size())
