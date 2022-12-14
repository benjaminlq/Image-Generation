"""Models module
"""
from .vae import BaseVAE, DeepVAE, ConvVAE

__all__ = ["BaseVAE", "DeepVAE", "ConvVAE"]

models = {"BaseVAE": BaseVAE,
          "DeepVAE": DeepVAE,
         # "ConvVAE": ConvVAE
          }