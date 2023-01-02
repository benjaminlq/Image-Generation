"""CIFAR10 used for generation project
"""

from pathlib import Path
from typing import Union

from torchvision import datasets, transforms

import config
from dataloaders import BaseDataLoader


class CIFARDataLoader(BaseDataLoader):
    """CIFAR10 Digit DataLoader"""

    def __init__(
        self,
        data_path: Union[str, Path] = config.DATA_PATH,
        batch_size: int = 32,
        image_size: int = 28,
        std_normalize: bool = False,
    ):
        """CIFAR10 Data Module

        Args:
            data_path (Union[str, Path], optional): Path to load/save CIFAR10 datasets.
            batch_size (int, optional): Batch Size. Defaults to 32.
        """
        super(CIFARDataLoader, self).__init__(data_path, batch_size)

        if std_normalize:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )

        self.train_dataset = datasets.CIFAR10(
            data_path, download=True, train=True, transform=train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            data_path, download=True, train=False, transform=test_transform
        )

    def __str__(self):
        """Dataset Name"""
        return "cifar10"


if __name__ == "__main__":
    data_manager = CIFARDataLoader()
    train_loader = data_manager.train_loader()
    images, labels = next(iter(train_loader))
    print("Train Batch images size:", images.size())
    print("Train Batch labels size:", len(labels))
    test_loader = data_manager.test_loader()
    samples = next(iter(test_loader))
    print(samples)
    print("Test Batch images size:", images.size())
    print("Test Batch labels size:", len(labels))
