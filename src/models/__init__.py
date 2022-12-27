"""Models module
"""
from .vae import BaseVAE, DeepVAE, ConvVAE
from .cvae import BaseCVAE, DeepCVAE, ConvCVAE

__all__ = ["BaseVAE", "DeepVAE", "ConvVAE", "BaseCVAE", "DeepCVAE", "ConvCVAE"]

models = {"BaseVAE": BaseVAE, "DeepVAE": DeepVAE, "ConvVAE": ConvVAE,
          "BaseCVAE": BaseCVAE, "DeepCVAE": DeepCVAE, "ConvCVAE": ConvCVAE}
