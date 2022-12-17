"""dataloaders module
"""
from .base import BaseDataLoader
from .mnist import MNISTDataLoader
from .fmnist import FashionMNISTDataLoader
from .cifar10 import CIFARDataLoader

__all__ = [
    "MNISTDataLoader",
    "BaseDataLoader",
    "FashionMNISTDataLoader",
    "CIFARDataLoader",
]

dataloaders = {
    "mnist": (MNISTDataLoader, (1, 28, 28)),
    "fmnist": (FashionMNISTDataLoader, (1, 28, 28)),
    "cifar10": (CIFARDataLoader, (3, 28, 28)),
}
