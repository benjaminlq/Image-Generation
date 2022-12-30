"""Utility Functions
"""
import os
import random
import argparse
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


def plot_images(imgs: Sequence, save_path=False):
    """Plot Images

    Args:
        imgs (Sequence): Sequence of images
    """
    _, axs = plt.subplots(nrows=1, ncols=len(imgs), squeeze=True)
    for i in range(len(imgs)):
        img = imgs[i].detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[i].imshow(np.asarray(img))
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def compare_recon(original_images: Sequence, recon_images: Sequence, save_path=False):
    """Plots to compare original images and reconstructed images

    Args:
        original_images (Sequence): Sequence of original torch images
        recon_images (Sequence): Sequence of reconstructed images
    """
    _, axs = plt.subplots(nrows=2, ncols=len(original_images), squeeze=True)

    for i in range(len(original_images)):
        origin_img = original_images[i].detach()
        recon_img = recon_images[i].detach()
        origin_img = torchvision.transforms.functional.to_pil_image(origin_img)
        recon_img = torchvision.transforms.functional.to_pil_image(recon_img)
        axs[0, i].imshow(np.asarray(origin_img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[1, i].imshow(np.asarray(recon_img))
        axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_loss(total_loss: list, recon_loss: list, kld_loss: list, save_path=False):
    """Plot Loss

    Args:
        total_loss (list): List of total loss
        recon_loss (list): List of reconstruction loss
        kld_loss (list): List of KL divergence loss
        save_path (bool, optional): Path to save img. Defaults to False.
    """
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(total_loss, color="red")
    plt.title("Total Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")

    plt.subplot(1, 3, 2)
    plt.plot(recon_loss, color="blue", label="recon")
    plt.title("Recon Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Recon Loss")

    plt.subplot(1, 3, 3)
    plt.plot(kld_loss, color="green", label="kld")
    plt.title("KLD Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("KLD Loss")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def get_argument_parser():
    """Input Parameters for training model
    - No of Epochs
    - Batch Size
    - Learning Rate
    - Model Type
    - Load Existing Checkpoint
    - Early Stopping Setting
    - Patience Epoch Count

    Returns:
        args: Arguments for training models
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        help="How many epochs you need to run (default: 10)",
        type=int,
        default=config.EPOCHS,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help="The number of images in a batch (default: 32)",
        type=int,
        default=config.BATCH_SIZE,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate used for optimizer (default: 1e-4)",
        type=float,
        default=config.LEARNING_RATE,
    )

    parser.add_argument(
        "-md",
        "--model",
        help="Type of model instance used for training",
        type=str,
        default="BaseVAE",
    )

    parser.add_argument(
        "-l", "--load", help="Load Model from previous ckpt", type=str, default=False
    )

    parser.add_argument(
        "-es",
        "--earlystop",
        help="Whether EarlyStop during model training",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-p",
        "--patience",
        help="How Many epoches for Early Stopping",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="Which Dataset to use",
        type=str,
        default="mnist",
    )

    parser.add_argument(
        "-ls",
        "--loss",
        help="Which Loss to use",
        type=str,
        default="mse",
    )
    
    parser.add_argument(
        "-hd",
        "--hidden",
        help="Length of latent vector",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    return args