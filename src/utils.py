"""Utility Functions
"""
import os
import random
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image

import config
from config import LOGGER
from models.gan import Generator


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


def plot_gan_loss(
    generator_loss: list, discriminator_loss: list, save_path: Optional[str] = None
):
    """Plot Loss Function for GAN training

    Args:
        generator_loss (list): Generator Loss History
        discriminator_loss (list): Discriminator Loss History
        save_path (Optional[str], optional): Path to save img. Defaults to None.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(generator_loss, color="red")
    plt.title("Generator Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")

    plt.subplot(1, 2, 2)
    plt.plot(discriminator_loss, color="blue")
    plt.title("Discriminator Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Discriminator Loss")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def sample_gan_image(generator: Generator, save_path: str, n_row: int = 8):
    """Sample batch of random GAN images

    Args:
        generator (Generator): Generator Module
        save_path (str): Path to save image
        n_row (int, optional): Number of rows to per conditional class. Defaults to 8.
    """
    z = torch.randn(
        (n_row * generator.num_classes, generator.hidden_size), device=config.DEVICE
    )
    if generator.conditional:
        labels = torch.tensor(
            [num for _ in range(n_row) for num in range(generator.num_classes)],
            dtype=torch.int32,
            device=config.DEVICE,
        )
    else:
        labels = None
    gen_imgs = generator(z, labels)
    save_image(gen_imgs, save_path, nrow=generator.num_classes, normalize=True)
