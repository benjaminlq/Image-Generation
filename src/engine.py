"""Train & Eval functions
"""
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import utils
from config import LOGGER
from models.gan import Discriminator, Generator


def BCE_VAE_loss(
    x_recon: torch.tensor,
    x: torch.tensor,
    mu: torch.tensor,
    log_var: torch.tensor,
    alpha: float = config.ALPHA,
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
    BCE = (
        F.binary_cross_entropy(x_recon.view(bs, -1), x.view(bs, -1), reduction="none")
        .sum(dim=1)
        .mean(dim=0)
    )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean(dim=0)
    return BCE * alpha + KLD, BCE, KLD


def MSE_VAE_loss(
    x_recon: torch.tensor,
    x: torch.tensor,
    mu: torch.tensor,
    log_var: torch.tensor,
    alpha: float = config.ALPHA,
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
    data_manager: Callable,
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
        data_manager(Callable): Data Manager used for training.
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
    train_loader = data_manager.train_loader()
    val_loader = data_manager.test_loader()

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
        for batch_idx, (images, labels) in enumerate(tk0):
            images = images.to(config.DEVICE)
            optimizer.zero_grad()
            if hasattr(model, "input_embedding"):
                labels = labels.to(config.DEVICE)
                recon_batch, mu, log_var = model(images, labels)
            else:
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
            model, loss_function, val_loader, epoch, str(data_manager)
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
                ckpt_path = str(
                    model_path
                    / f"{str(model)}_{model.hidden_size}_{str(data_manager)}.pt"
                )
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
        if not model_path.exists():
            model_path.mkdir(parents=True)
        history_path = str(
            model_path / f"history_{str(model.hidden_size)}_{str(data_manager)}.png"
        )
        utils.plot_loss(
            history["total_loss"],
            history["recon_loss"],
            history["kld_loss"],
            history_path,
        )
    return history


def eval(
    model: Callable,
    loss_function: Callable,
    val_loader: DataLoader,
    epoch: int = 0,
    dataset: str = "mnist",
):
    """Evaluate Function Engine

    Args:
        model (Callable): Model to be evaluated
        loss_function (Callable): Custom Loss Function to use for VAE training. Should consist of a reconstruction loss and regularization loss.
        val_loader (DataLoader): Validation DataLoader
        epoch (int, optional): Current Epoch on training loop. Defaults to 0.
        dataset (str): Dataset used for evaluation. Default to mnist.

    Returns:
        tuple: total_epoch_loss, recon_epoch_loss, kld_epoch_loss
    """
    model.eval()
    with torch.no_grad():
        epoch_loss, recon_epoch_loss, kld_epoch_loss = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(config.DEVICE)
            if hasattr(model, "input_embedding"):
                labels = labels.to(config.DEVICE)
                recon_batch, mu, log_var = model(images, labels)
            else:
                recon_batch, mu, log_var = model(images)
            loss, RECON, KLD = loss_function(recon_batch, images, mu, log_var)
            epoch_loss += loss.item()
            recon_epoch_loss += RECON.item()
            kld_epoch_loss += KLD.item()

            if batch_idx == 0:
                n = min(images.size(0), 8)

                img_path = config.ARTIFACT_PATH / "model_ckpt" / str(model) / "images"
                if not img_path.exists():
                    img_path.mkdir(parents=True)

                ## Reconstruction
                origin_imgs = images[:n].cpu()
                recon_imgs = recon_batch[:n].cpu()
                utils.compare_recon(
                    origin_imgs,
                    recon_imgs,
                    save_path=str(
                        img_path
                        / f"reconstruction_{str(model.hidden_size)}_{dataset}_{str(epoch)}.png"
                    ),
                )

                ## Generation
                if hasattr(model, "input_embedding"):
                    all_labels = torch.tensor(
                        range(model.num_classes), dtype=torch.int32
                    )
                    zs = torch.randn((model.num_classes, model.hidden_size))
                    generated_imgs = model.decode(
                        zs.to(config.DEVICE), all_labels.to(config.DEVICE)
                    ).cpu()
                    utils.plot_images(
                        generated_imgs,
                        save_path=str(
                            img_path
                            / f"generation_{str(model.hidden_size)}_{dataset}_{str(epoch)}.png"
                        ),
                    )

    return (
        epoch_loss / len(val_loader),
        recon_epoch_loss / len(val_loader),
        kld_epoch_loss / len(val_loader),
    )


def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    data_manager: Callable,
    loss_function: Literal["bce", "mse"] = "bce",
    no_epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    k: int = 1,
    save: bool = True,
    load: bool = False,
):
    """Function for training GAN

    Args:
        generator (Generator): Generator to train
        discriminator (Discriminator): Discriminator to train
        data_manager(Callable): Data Manager used for training
        loss_function (Literal[bce|mse], optional): Loss Function for GAN training. If bce, used Binary Cross Entropy Loss for
            discriminator (similar to Ian Goodfellow paper). If MSE, use MSELoss to calculate error (Least Square GAN). Defaults to "bce".
        no_epochs (int, optional): _description_. Defaults to config.EPOCHS.
        learning_rate (float, optional): _description_. Defaults to config.LEARNING_RATE.
        k (int, optional): No of discriminator updates per generator update. Defaults to 1.
        save (bool, optional): Save model. Defaults to True.
        load (bool, optional): Load model from checkpoint before training. Defaults to False.

    Returns:
        dict: Generator Loss and Discriminator Loss
    """
    assert (
        generator.conditional == discriminator.conditional
    ), "Generator and Discriminator Conditional must match"
    conditional = generator.conditional

    optimizer_G = torch.optim.Adam(
        params=generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        params=discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    if loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    train_loader = data_manager.train_loader()
    generator.to(config.DEVICE)
    discriminator.to(config.DEVICE)
    LOGGER.info(f"Training Model {str(generator)} on {config.DEVICE}.")

    if load:
        generator_path = (
            config.ARTIFACT_PATH
            / "model_ckpt"
            / str(generator)
            / f"{str(generator)}.pt"
        )
        discriminator_path = (
            config.ARTIFACT_PATH
            / "model_ckpt"
            / str(discriminator)
            / f"{str(discriminator)}.pt"
        )
        if not (generator_path.exists() or discriminator.exists()):
            LOGGER.warning(
                f"Ckpt_path for {str(generator)} does not exist. Training New Model"
            )
        else:
            utils.load_model(generator, generator_path)
            utils.load_model(discriminator, discriminator_path)

    model_path = config.ARTIFACT_PATH / "model_ckpt" / str(generator)
    loss_history = {"generator": [], "discriminator": []}

    for epoch in range(no_epochs):
        generator.train()
        discriminator.train()

        tk0 = tqdm(train_loader, total=len(train_loader))
        for batch_idx, (real_imgs, real_labels) in enumerate(tk0):
            # Generate 1 batch of real images
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(config.DEVICE)
            real_labels = real_labels.to(config.DEVICE)

            for _ in range(k):

                # Generate 1 batch of fake images
                zs = torch.randn((bs, generator.hidden_size), device=config.DEVICE)
                if conditional:
                    gen_labels = torch.randint(
                        0, generator.num_classes, size=(bs,), device=config.DEVICE
                    )
                    gen_imgs = generator(zs, gen_labels)
                else:
                    gen_imgs = generator(zs)

                # Train Discriminator

                optimizer_D.zero_grad()
                real_targets = torch.ones(
                    (bs, 1), requires_grad=False, device=config.DEVICE
                )
                fake_targets = torch.zeros(
                    (bs, 1), requires_grad=False, device=config.DEVICE
                )
                if conditional:
                    real_probs = discriminator(real_imgs, real_labels)
                    fake_probs = discriminator(gen_imgs, gen_labels)
                else:
                    real_probs = discriminator(real_imgs)
                    fake_probs = discriminator(gen_imgs)

                real_loss_D = criterion(real_probs, real_targets)
                fake_loss_D = criterion(fake_probs, fake_targets)

                d_loss = (real_loss_D + fake_loss_D) / 2
                d_loss.backward()
                optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            zs = torch.randn((bs, generator.hidden_size), device=config.DEVICE)
            if conditional:
                gen_labels = torch.randint(
                    0, generator.num_classes, size=(bs,), device=config.DEVICE
                )
                gen_imgs = generator(zs, gen_labels)
                fake_probs = discriminator(gen_imgs, gen_labels)
            else:
                gen_imgs = generator(zs)
                fake_probs = discriminator(gen_imgs)

            fake_targets = torch.ones(
                (bs, 1), requires_grad=False, device=config.DEVICE
            )

            g_loss = criterion(fake_probs, fake_targets)
            g_loss.backward()
            optimizer_G.step()

            if (batch_idx + 1) % 200 == 0:
                print(
                    f"Epoch {epoch+1} - batch {batch_idx+1}: Generator Loss = {g_loss.item()}, Discriminator Loss = {d_loss.item()}"
                )

        LOGGER.info(
            f"Epoch {epoch+1}: Generator Loss = {g_loss.item()}, Discriminator Loss = {d_loss.item()}"
        )
        if save:
            img_path = model_path / "images"
            if not img_path.exists():
                img_path.mkdir(parents=True)
            utils.sample_gan_image(
                generator,
                str(
                    img_path
                    / f"gan_{generator.hidden_size}_{str(data_manager)}_{epoch}.png"
                ),
            )
            utils.save_model(
                generator,
                str(
                    model_path
                    / f"{str(generator)}_{generator.hidden_size}_{str(data_manager)}.pt"
                ),
            )
            utils.save_model(
                discriminator,
                str(
                    model_path
                    / f"{str(discriminator)}_{generator.hidden_size}_{str(data_manager)}.pt"
                ),
            )

        loss_history["generator"].append(g_loss.item())
        loss_history["discriminator"].append(d_loss.item())

    if save:
        if not model_path.exists():
            model_path.mkdir(parents=True)
        history_path = str(
            model_path / f"History_{str(generator.hidden_size)}_{str(data_manager)}.png"
        )
        utils.plot_gan_loss(
            loss_history["generator"], loss_history["discriminator"], history_path
        )

    return loss_history
