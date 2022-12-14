from .base import BaseDataLoader
from .mnist import MNISTDataLoader

__all__ = ['MNISTDataLoader', "BaseDataLoader"]

dataloaders = {"mnist":MNISTDataLoader}