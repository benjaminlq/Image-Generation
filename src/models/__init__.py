"""Models module
"""
from .vae import BaseVAE, DeepVAE, ConvVAE
from .cvae import BaseCVAE, DeepCVAE, ConvCVAE
from .gan import Generator, Discriminator

__all__ = ["BaseVAE", "DeepVAE", "ConvVAE", "BaseCVAE", "DeepCVAE", "ConvCVAE", "Generator", "Discriminator"]

models = {"BaseVAE": BaseVAE, "DeepVAE": DeepVAE, "ConvVAE": ConvVAE,
          "BaseCVAE": BaseCVAE, "DeepCVAE": DeepCVAE, "ConvCVAE": ConvCVAE}
gan_models = {"Generator":Generator, "Discriminator":Discriminator}