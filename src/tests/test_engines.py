from engine import train, BCE_VAE_loss, MSE_VAE_loss
from models import BaseVAE
from dataloaders import dataloaders
import torch

def test_BCE_loss():
    alpha = 100
    x = torch.rand(5,3,28,28)
    x_recon = torch.rand(5,3,28,28)
    mu = torch.rand(5,20)
    log_var = torch.rand(5,20)
    total_loss, bce, kld = BCE_VAE_loss(x_recon, x, mu, log_var, alpha = alpha)
    assert total_loss == bce * alpha + kld, "Inc"
    
def test_mse_loss():
    alpha = 100
    x = torch.rand(5,3,28,28)
    x_recon = torch.rand(5,3,28,28)
    mu = torch.rand(5,20)
    log_var = torch.rand(5,20)
    total_loss, mse, kld = MSE_VAE_loss(x_recon, x, mu, log_var, alpha = alpha)
    assert total_loss == mse * alpha + kld

def test_train():
    datamodule, img_size = dataloaders["mnist"]
    data_manager = datamodule(batch_size=1000)
    train_loader = data_manager.train_loader()
    test_loader = data_manager.test_loader()
    model = BaseVAE(input_size=img_size)
    
    train(model, BCE_VAE_loss, train_loader, test_loader, no_epochs=1, save=False)
    train(model, MSE_VAE_loss, train_loader, test_loader, no_epochs=1, save=False)
    
if __name__ == "__main__":
    # test_BCE_loss()
    # test_mse_loss()
    test_train()