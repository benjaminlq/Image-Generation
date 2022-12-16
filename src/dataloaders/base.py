"""Base DataLoader
"""

from pathlib import Path
from typing import Callable, Optional, Union

from torch.utils.data import DataLoader, Dataset

import config


class BaseDataLoader:
    """Base DataLoader Module"""

    def __init__(
        self,
        data_path: Union[str, Path] = config.DATA_PATH,
        batch_size: int = 32,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
    ):
        """MNIST Data Module

        Args:
            data_path (Union[str, Path], optional): Path to load/save MNIST datasets.
            batch_size (int, optional): Batch Size. Defaults to 32.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

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
