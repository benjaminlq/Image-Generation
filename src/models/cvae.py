"""Models using Variational Auto Encoder
"""
from typing import Literal

import torch
import torch.nn as nn

import config
from models.block import *
from models.vae import BaseVAE, ConvVAE

class BaseCVAE(BaseVAE):
    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 256,
        hidden_size: int = config.HIDDEN_SIZE,
        cond_size: int = 128,
        num_classes: int = 10,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
        **kwargs,
    ):
        super(BaseCVAE, self).__init__(input_size=input_size, common_size=common_size, hidden_size=hidden_size, activation=activation, **kwargs)
        
        self.cond_size = cond_size
        self.num_classes = num_classes
        self.input_embedding = nn.Embedding(num_classes, cond_size)

        self.mean_fc = nn.Linear(common_size + cond_size, hidden_size)
        self.var_fc = nn.Linear(common_size + cond_size, hidden_size)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            MLPBlock(self.c * self.h * self.w, common_size)
        )
        self.decoder = nn.Sequential(
            MLPBlock(hidden_size + cond_size, common_size),
            nn.Linear(common_size, self.c * self.h * self.w),
            nn.Unflatten(1, (self.c, self.h, self.w)),
        )


    def encode(self, input_imgs: torch.tensor, input_classes: torch.tensor) -> torch.tensor:
        """Encoder converts input images and input classes to Conditional probability distribution (Gaussian) of latent vectors P(z|x, y)

        Args:
            input_imgs (torch.tensor): Input Images
            input_classes (torch.tensor): Input Classes of input images

        Returns:
            torch.tensor: Mean and Diagonal Covariance Matrix of Gaussian describing the probability P(z|x, y)
        """
        # Input_imgs: (bs, c, h, w). Input_classes: (bs, )
        y = self.input_embedding(input_classes) # (bs, cond_size)
        x = self.encoder(input_imgs)
        x_cond = torch.cat((x, y), dim = 1)
        mu = self.mean_fc(x_cond)
        log_var = self.var_fc(x_cond)
        return mu, log_var

    def decode(self, z: torch.tensor, input_classes: torch.tensor) -> torch.tensor:
        """Reconstruction of latent vector z into images

        Args:
            z (torch.tensor): Latent vector z
            input_classes (torch.tensor): Input Classes for image reconstruction
  
        Returns:
            torch.tensor: Reconstructed Image based on input class. P(x_hat | z, y)
        """
        bs = z.size(0)
        y = self.input_embedding(input_classes)
        cond_z = torch.cat((z, y), dim = 1)
        out = self.activation(self.decoder(cond_z))
        return out.view(bs, self.c, self.h, self.w)

    def forward(self, input_imgs: torch.tensor, input_classes: torch.tensor):
        """Forward Propagation. The steps involved in a forward pass:
        1. Given input image x and input class y, estimate the Gaussian distribution of latent representation
        P(z|x, y) ~ N(mu(z|x,y), std(z|x,y))
        2. Based P(z|x, y), sample a latent vector z
        3. Reconstruct image x_hat based on latent vector z and input classes y (Decode)

        Args:   
            input_imgs (torch.tensor): Input Images
            input_classes (torch.tensor): Input Classes of input images

        Returns:
            tuple(torch.tensor, torch.tensor, torch.tensor): reconstructed images, latent mean, latent std 
        """

        mu, log_var = self.encode(input_imgs, input_classes)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z, input_classes)
        return out, mu, log_var

    def __str__(self):
        """Model Name"""
        return "BaseCVAE"
    
class DeepCVAE(BaseCVAE):
    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 256,
        hidden_size: int = config.HIDDEN_SIZE,
        cond_size: int = 128,
        num_classes: int = 10,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
    ):  
    
        super(DeepCVAE, self).__init__(input_size, common_size, hidden_size, cond_size, num_classes, activation)
        
        self.mean_fc = nn.Sequential(
            MLPBlock(common_size + cond_size, 256),
            nn.Linear(256, hidden_size)
        )
        self.var_fc = nn.Sequential(
            MLPBlock(common_size + cond_size, 256),
            nn.Linear(256, hidden_size)
        )
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            MLPBlock(self.c * self.h * self.w, 784),
            MLPBlock(784, 392),
            MLPBlock(392, common_size),
        )

        self.decoder = nn.Sequential(
            MLPBlock(self.hidden_size + cond_size, 256),
            MLPBlock(256, 392),
            MLPBlock(392, 784),
            nn.Linear(784, self.c * self.h * self.w),
            nn.Unflatten(1, (self.c, self.h, self.w)),
        )
        
    def __str__(self):
        """Model Name"""
        return "DeepCVAE"
        
class ConvCVAE(BaseCVAE, ConvVAE):
    def __init__(
        self,
        input_size: tuple = (1, 28, 28),
        common_size: int = 256,
        hidden_size: int = config.HIDDEN_SIZE,
        cond_size: int = 128,
        num_classes: int = 10,
        activation: Literal["Tanh", "Sigmoid"] = "Sigmoid",
        kernel_size: int = 3,
    ):  
    
        super(ConvCVAE, self).__init__(input_size=input_size, common_size=common_size, hidden_size=hidden_size, cond_size=cond_size,
                                       num_classes=num_classes, activation=activation, kernel_size=kernel_size)
        
        self.mean_fc = nn.Sequential(
            MLPBlock(common_size + cond_size, 256),
            nn.Linear(256, hidden_size)
        )
        self.var_fc = nn.Sequential(
            MLPBlock(common_size + cond_size, 256),
            nn.Linear(256, hidden_size)
        )
        
        ## Input = c, h, w
        self.encoder = nn.Sequential(
            ConvBlock(self.c, 16, kernel_size=self.kernel_size, padding=self.padding),  # 16, h, w
            DownSample(16, 32, kernel_size=self.kernel_size, padding=self.padding),  # 32, h/2, w/2
            DownSample(32, 64, kernel_size=self.kernel_size, padding=self.padding),  # 64, h/4, w/4
            nn.Flatten(),  # 64 & (h/4) * (w/4)
            MLPBlock(64 * self.final_height * self.final_width, 784),
            MLPBlock(784, 392),
            MLPBlock(392, common_size),
        )  # Common Size

        ## Input = hidden_size
        self.decoder = nn.Sequential(
            MLPBlock(hidden_size + cond_size, common_size),
            MLPBlock(common_size, 392),
            MLPBlock(392, 784),
            MLPBlock(784, 64 * self.final_height * self.final_width),
            nn.Unflatten(1, (64, self.final_height, self.final_width)),  # 64 & (h/4) * (w/4)
            UpSample(64, 32, kernel_size=self.kernel_size, padding=self.padding),
            UpSample(32, 16, kernel_size=self.kernel_size, padding=self.padding),
            ConvBlock(16, self.c, kernel_size=self.kernel_size, padding=self.padding),
        )
        
    def __str__(self):
        """Model Name"""
        return "ConvCVAE"
    
if __name__ == "__main__":
    sample_imgs = torch.randn((5, 3, 28, 28))
    c, h, w = sample_imgs.size(1), sample_imgs.size(2), sample_imgs.size(3)
    model = ConvCVAE(input_size=(c, h, w))
    sample_labels = torch.randint(low=0, high=9, size=(5,))
    out, mu, log_var = model(sample_imgs, sample_labels)
    print("Reconstruced Images Batch Size:", out.size())
    print("Mu Batch Size:", mu.size())
    print("Log Var Batch Size:", log_var.size())
    