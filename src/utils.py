"""Utility Functions
"""
import os
import random
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import config
from config import LOGGER


def save_model(model, path: str):
    """Save Model

    Args:
        model (Callable): Model
        path (str): Path to save ckpt
    """
    torch.save(model.state_dict(), path)
    LOGGER.info(f"Model {str(model)} saved successfully at {path}")


def load_model(model, path: str):
    """Load Model

    Args:
        model (Callable): Model
        path (str): Path to load ckpt
    """
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    LOGGER.info(f"Model {str(model)} loaded successfully from {path}")


def seed_everything(seed: int = 2023):
    """Set seed for reproducability"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_images(imgs: Sequence):
    """Plot Images

    Args:
        imgs (Sequence): Sequence of images
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(imgs), squeeze=True)
    for i in range(len(imgs)):
        img = imgs[i].detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def compare_recon(original_images: Sequence, recon_images: Sequence):
    """Plots to compare original images and reconstructed images

    Args:
        original_images (Sequence): Sequence of original torch images
        recon_images (Sequence): Sequence of reconstructed images
    """
    fig, axs = plt.subplots(nrows=2, ncols=len(original_images), squeeze=True)
    for i in range(len(original_images)):
        origin_img = original_images[i].detach()
        recon_img = recon_images[i].detach()
        origin_img = torchvision.transforms.functional.to_pil_image(origin_img)
        recon_img = torchvision.transforms.functional.to_pil_image(recon_img)
        axs[0, i].imshow(np.asarray(origin_img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[1, i].imshow(np.asarray(recon_img))
        axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
