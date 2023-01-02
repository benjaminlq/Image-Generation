"""Unit Tests for Engine functions
"""
import torch

from dataloaders import dataloaders
from engine import BCE_VAE_loss, MSE_VAE_loss, train, train_gan
from models import BaseVAE
from models import Generator, Discriminator

def test_BCE_loss():
    """Test Behaviour of BCE loss"""
    alpha = 100
    x = torch.rand(5, 3, 28, 28)
    x_recon = torch.rand(5, 3, 28, 28)
    mu = torch.rand(5, 20)
    log_var = torch.rand(5, 20)
    total_loss, bce, kld = BCE_VAE_loss(x_recon, x, mu, log_var, alpha=alpha)
    assert total_loss == bce * alpha + kld, "Inc"


def test_mse_loss():
    """Test Behaviour of BCE loss"""
    alpha = 100
    x = torch.rand(5, 3, 28, 28)
    x_recon = torch.rand(5, 3, 28, 28)
    mu = torch.rand(5, 20)
    log_var = torch.rand(5, 20)
    total_loss, mse, kld = MSE_VAE_loss(x_recon, x, mu, log_var, alpha=alpha)
    assert total_loss == mse * alpha + kld


def test_train():
    """Test Train and Evaluation engine functions"""
    datamodule, img_size = dataloaders["cifar10"]
    data_manager = datamodule(batch_size=1000)
    model = BaseVAE(input_size=img_size)

    train(model, BCE_VAE_loss, data_manager, no_epochs=1, save=False)
    train(model, MSE_VAE_loss, data_manager, no_epochs=1, save=False)

def test_train_GAN():
    datamodule, img_size = dataloaders["mnist"]
    data_manager = datamodule(batch_size=250)
    generator = Generator(img_shape=img_size)
    discriminator = Discriminator(img_shape=img_size)
    train_gan(generator, discriminator, data_manager, no_epochs=1, save=False)
    train_gan(generator, discriminator, data_manager, loss_function = "mse", no_epochs=1, save=False)
    cgenerator = Generator(img_shape=img_size, conditional=True)
    cdiscriminator = Discriminator(img_shape=img_size, conditional=True)
    train_gan(cgenerator, cdiscriminator, data_manager, no_epochs=1, save=False)
    
if __name__ == "__main__":
    test_train_GAN()
