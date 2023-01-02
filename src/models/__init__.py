"""Models module
"""
from .vae import BaseVAE, DeepVAE, ConvVAE
from .cvae import BaseCVAE, DeepCVAE, ConvCVAE
from .gan import (
    Generator,
    Discriminator,
    CGenerator,
    CDiscriminator,
    ConvGenerator,
    ConvDiscriminator,
)

__all__ = [
    "BaseVAE",
    "DeepVAE",
    "ConvVAE",
    "BaseCVAE",
    "DeepCVAE",
    "ConvCVAE",
    "Generator",
    "Discriminator",
    "CGenerator",
    "CDiscriminator",
    "ConvGenerator",
    "ConvDiscriminator",
]

models = {
    "BaseVAE": BaseVAE,
    "DeepVAE": DeepVAE,
    "ConvVAE": ConvVAE,
    "BaseCVAE": BaseCVAE,
    "DeepCVAE": DeepCVAE,
    "ConvCVAE": ConvCVAE,
    "GAN": Generator,
    "CGAN": CGenerator,
    "ConvGAN": ConvGenerator,
    "Discriminator": Discriminator,
    "CDiscriminator": CDiscriminator,
    "ConvDiscriminator": ConvDiscriminator,
}
