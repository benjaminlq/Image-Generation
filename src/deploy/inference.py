"""Inference Module for Deployment
"""
import json
import random
from glob import glob
from pathlib import Path

from torchvision import datasets, transforms

import config
import utils
from dataloaders import dataloaders
from models import models


class Inference:
    """Inference Module"""

    def __init__(
        self,
        artifact_path: Path = config.ARTIFACT_PATH,
        data_path: Path = config.DATA_PATH,
    ):
        """Inference Instance

        Args:
            artifact_path (Path, optional): Path to artifacts to access model checkpoints. Defaults to config.ARTIFACT_PATH.
            data_path (Path, optional): Path to dataset to access images. Defaults to config.DATA_PATH.
        """
        model_paths = glob(
            str(artifact_path / "model_ckpt" / "**" / "*.pt"), recursive=True
        )
        self.model_dict = {}
        for model_path in model_paths:
            model_name = model_path.split("/")[-1][:-3]
            model_type, hidden_size, dataset = model_name.split("_")
            hidden_size = int(hidden_size)
            _, input_size = dataloaders[dataset]
            self.model_dict[model_name] = models[model_type](
                input_size=input_size, hidden_size=hidden_size
            )
            self.model_dict[model_name].eval()
            utils.load_model(self.model_dict[model_name], model_path)

        print(self.model_dict.keys())

        infer_transforms = transforms.Compose(
            [transforms.RandomCrop(28, padding=4), transforms.ToTensor()]
        )

        self.datasets = {
            "mnist": datasets.MNIST(
                data_path, download=True, train=True, transform=infer_transforms
            )
        }

        self.index_dict = {}
        for dataset in self.datasets.keys():
            with open(str(config.DEPLOY_PATH / "jsons" / f"{dataset}.json"), "r") as f:
                self.index_dict[dataset] = json.load(f)

    def reconstruction(
        self, class_no: int, model_type: str, hidden_size: int, dataset: str
    ):
        """Reconstruction Feature. Take a random image of a specified class from the dataset, generate lalent vector distribution and
        reconstruct using Decoder. Quality of reconstruction is based on model type and hidden size of latent vector.

        Args:
            class_no (int): Class to sample images from.
            model_type (str): Model checkpoint to load for reconstruction process
            hidden_size (int): Number of dimensions (features) of Latent Vector Space
            dataset (str): Dataset used for inference

        Returns:
            Tuple(torch.tensor, torch.tensor): Target Image, Reconstructed Image
        """
        model_name = "_".join([model_type, str(hidden_size), dataset])
        img_idx = random.choice(self.index_dict[dataset][str(class_no)])
        target_img, _ = self.datasets[dataset][img_idx]

        recon_img, _, _ = self.model_dict[model_name](
            target_img.unsqueeze(0).to(config.DEVICE)
        )

        return target_img, recon_img.squeeze(0).cpu()


if __name__ == "__main__":
    infer_cls = Inference()
    target_img, recon_img = infer_cls.reconstruction(3, "ConvVAE", 2, "mnist")
    print(target_img.size())
    print(recon_img.size())
