"""Train & Eval functions
"""
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
import utils
from config import LOGGER


def BCE_VAE_loss(
    x_recon: torch.tensor, x: torch.tensor, mu: torch.tensor, log_var: torch.tensor, alpha: float = config.ALPHA,
):
    """Loss for Variational Auto Encoder. Reconstruction loss used is Binary Cross Entropy (BCE) on each pixel.

    Args:
        x_recon (torch.tensor): Reconstructed Images
        x (torch.tensor): Original Images
        mu (torch.tensor): Mean of Gaussian Distribution of Latent vector
        log_var (torch.tensor): Variance of Gaussian Distribution of Latent vector
        alpha (float): Weight to balance between reconstruction loss and KL divergence. Higher weight means higher emphasis on reconstruction loss.
        Default to 1000.
        
    Returns:
        tuple: total_loss, bce_loss, kld_loss
    """
    bs = x.size(0)
    BCE = F.binary_cross_entropy(x_recon.view(bs, -1), x.view(bs, -1), reduction="none").sum(dim=1).mean(dim=0)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean(dim=0)
    return BCE * alpha + KLD, BCE, KLD


def MSE_VAE_loss(
    x_recon: torch.tensor, x: torch.tensor, mu: torch.tensor, log_var: torch.tensor, alpha: float = config.ALPHA,
):
    """Loss for Variational Auto Encoder. Reconstruction loss used is Mean Squared Error (MSE) on each pixel.

    Args:
        x_recon (torch.tensor): Reconstructed Images
        x (torch.tensor): Original Images
        mu (torch.tensor): Mean of Gaussian Distribution of Latent vector
        log_var (torch.tensor): Variance of Gaussian Distribution of Latent vector
        alpha (float): Weight to balance between reconstruction loss and KL divergence. Higher weight means higher emphasis on reconstruction loss.
        Default to 1000.

    Returns:
        tuple: total_loss, mse_loss, kld_loss
    """
    bs = x.size(0)
    MSE = F.mse_loss(x_recon.view(bs, -1), x.view(bs, -1), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean(0)
    return MSE * alpha + KLD, MSE, KLD


def train(
    model: Callable,
    loss_function: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    no_epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    early_stopping: bool = False,
    patience: int = 5,
    save: bool = True,
    load: bool = False,
) -> dict:
    """Training Function

    Args:
        model (Callable): Model to train
        loss_function (Callable): Custom Loss Function to use for VAE training. Should consist of a reconstruction loss and regularization loss.
        train_loader (DataLoader): Training Dataloader
        val_loader (DataLoader): Validation Dataloader
        no_epochs (int): No of epochs
        learning_rate (float): Learning Rate
        early_stopping (bool, optional): Whether to use EarlyStopping. Defaults to False.
        patience (int, optional): How many epochs without val_acc improvement for EarlyStopping. Defaults to 5.
        save (bool, optional): Save model. Defaults to True.
        load (bool, optional): Load model from checkpoint before training. Defaults to False.

    Returns:
        dict: Total, Recon & KLD loss across epoches
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )

    model.to(config.DEVICE)
    LOGGER.info(f"Training Model on {config.DEVICE}")

    if load:
        ckpt_path = config.ARTIFACT_PATH / "model_ckpt" / str(model) / "model.pt"
        if not ckpt_path.exists():
            LOGGER.warning(
                f"Ckpt_path for {str(model)} does not exist. Training New Model"
            )
        else:
            utils.load_model(model, ckpt_path)

    best_total = float("inf")
    patience_count = 0
    history = {"total_loss": [], "recon_loss": [], "kld_loss": []}

    model_path = config.ARTIFACT_PATH / "model_ckpt" / str(model)

    for epoch in range(no_epochs):
        model.train()
        epoch_loss, recon_epoch_loss, kld_epoch_loss = 0, 0, 0
        tk0 = tqdm(train_loader, total=len(train_loader))
        for batch_idx, (images, _) in enumerate(tk0):
            images = images.to(config.DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(images)
            loss, RECON, KLD = loss_function(recon_batch, images, mu, log_var)

            loss.backward()
            epoch_loss += loss.item()
            recon_epoch_loss += RECON.item()
            kld_epoch_loss += KLD.item()

            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            optimizer.step()
            if (batch_idx + 1) % 1000 == 0:
                print(
                    f"Train Epoch {epoch + 1} - Total Loss = {loss.item():.4f}, Reconstruction Loss = {RECON.item():.4f}, KLD Loss = {KLD.item():.4f}"
                )

        LOGGER.info(
            f"====> Epoch {epoch + 1} Total Loss = {epoch_loss/len(train_loader):.4f}, Reconstruction Loss = {recon_epoch_loss/len(train_loader):.4f}, KLD Loss =  {kld_epoch_loss/len(train_loader):.4f}"
        )

        val_total_loss, val_recon_loss, val_kld_loss = eval(
            model, loss_function, val_loader, epoch
        )
        scheduler.step(val_total_loss)
        history["total_loss"].append(val_total_loss)
        history["recon_loss"].append(val_recon_loss)
        history["kld_loss"].append(val_kld_loss)

        if val_total_loss < best_total:
            LOGGER.info(
                f"{str(model)}: Val Loss improved at epoch {epoch + 1} from {best_total} to {val_total_loss}"
            )
            best_total = val_total_loss
            patience_count = 0
            if save:
                if not model_path.exists():
                    model_path.mkdir(parents=True)
                ckpt_path = str(model_path / "model.pt")
                utils.save_model(model, ckpt_path)

        else:
            LOGGER.info(
                f"{str(model)}: Validation Loss from epoch {epoch + 1} did not improve"
            )
            patience_count += 1
            if early_stopping and patience_count == patience:
                LOGGER.warning(
                    f"{str(model)}: No val loss improvement for {patience} consecutive epochs. Early Stopped at epoch {epoch + 1}"
                )

    if save: 
        history_path = str(model_path / "history.png")
        utils.plot_images(history_path)
    return history


def eval(
    model: Callable, loss_function: Callable, val_loader: DataLoader, epoch: int = 0
):
    """Evaluate Function Engine

    Args:
        model (Callable): Model to be evaluated
        loss_function (Callable): Custom Loss Function to use for VAE training. Should consist of a reconstruction loss and regularization loss.
        val_loader (DataLoader): Validation DataLoader
        epoch (int, optional): Current Epoch on training loop. Defaults to 0.

    Returns:
        tuple: total_epoch_loss, recon_epoch_loss, kld_epoch_loss
    """
    model.eval()
    with torch.no_grad():
        epoch_loss, recon_epoch_loss, kld_epoch_loss = 0, 0, 0
        for batch_idx, (images, _) in enumerate(val_loader):
            images = images.to(config.DEVICE)
            recon_batch, mu, log_var = model(images)
            loss, RECON, KLD = loss_function(recon_batch, images, mu, log_var)
            epoch_loss += loss.item()
            recon_epoch_loss += RECON.item()
            kld_epoch_loss += KLD.item()

            if batch_idx == 0:
                n = min(images.size(0), 8)
                comparison = torch.cat(
                    [images[:n], recon_batch.view(images.size(0), 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    str(config.ARTIFACT_PATH / f"reconstruction_{str(epoch)}.png"),
                    nrow=n,
                )

    return (
        epoch_loss / len(val_loader),
        recon_epoch_loss / len(val_loader),
        kld_epoch_loss / len(val_loader),
    )
