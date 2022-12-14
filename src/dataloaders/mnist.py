"""Datasets used for generation project
"""

from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


## Prepare datasets
class MNIST_Dataloader:
    """MNIST Digit Dataset"""

    def __init__(
        self,
        data_path: Union[str, Path] = config.DATA_PATH,
        batch_size: int = 32,
    ):
        """MNIST Data Module

        Args:
            data_path (Union[str, Path], optional): Path to load/save MNIST datasets.
            batch_size (int, optional): Batch Size. Defaults to 32.
        """
        self.data_path = data_path
        self.batch_size = batch_size

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)]
        )

        self.train_dataset = datasets.MNIST(
            data_path, download=True, train=True, transform=train_transform
        )
        self.test_dataset = datasets.MNIST(
            data_path, download=True, train=False, transform=test_transform
        )

    def train_loader(self) -> DataLoader:
        """Train DataLoader

        Returns:
            DataLoader: Train DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        """Test DataLoader

        Returns:
            DataLoader: Test DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )


if __name__ == "__main__":
    data_manager = MNIST_Dataloader()
    train_loader = data_manager.train_loader()
    images, labels = next(iter(train_loader))
    print("Batch images size:", images.size())
    print("Batch labels size:", len(labels))
    test_loader = data_manager.test_loader()
    samples = next(iter(test_loader))
    print(samples)
