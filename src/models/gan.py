"""GAN Model
"""
import torch
import torch.nn as nn

from typing import Optional, Union
from models.block import *
import config

class Generator(nn.Module):
    """Generator Module
    """
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        hidden_size: int = 64,
        conditional: bool = False,
        emb_size: int = 32,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ):
        """Generator Module

        Args:
            img_shape (tuple, optional): Size of images to be generated. Defaults to (3, 28, 28).
            hidden_size (int, optional): Hidden size of input noise latent vector to Generator. Defaults to 64.
            conditional (bool, optional): If True, input of both generator and discriminator includes conditional class. Defaults to False.
            cond_size (int, optional): _description_. Defaults to 32.
            num_classes (int, optional): _description_. Defaults to 10.
        """
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.img_shape = img_shape
        self.conditional = conditional
        self.linear_1 = MLPBlock(self.hidden_size, 128)
        self.num_classes = num_classes
        if self.conditional:
            self.emb_size = emb_size
            self.embedding = nn.Embedding(num_classes, emb_size)
            self.linear_1 = MLPBlock(hidden_size + emb_size, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.linear_2 = MLPBlock(128, 256)
        self.linear_3 = MLPBlock(256, 512)
        self.linear_4 = MLPBlock(512, 1024)
        self.linear_out = nn.Linear(1024, self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        self.tanh = nn.Tanh()
        
    def forward(self, z: torch.tensor, cond_class: Optional[Union[int, torch.tensor]] = None) -> torch.tensor:
        """Forward Pass

        Args:
            z (torch.tensor): Noise Latent vectors to feed to Generator
            cond_class (Optional[Union[int, list]], optional): Class or List of class for conditional GAN. Model P(z|c). Defaults to None.

        Returns:
            torch.tensor: Generated Images
        """
        if cond_class is not None:
            if isinstance(cond_class, int):
                cond_class = torch.tensor([cond_class], dtype = torch.int32)
            input_embs = self.embedding(cond_class.to(config.DEVICE))
            z = torch.cat((z, input_embs), dim = 1)
        
        x = self.dropout(self.linear_1(z))
        x = self.dropout(self.linear_2(x))
        x = self.dropout(self.linear_3(x))
        x = self.dropout(self.linear_4(x))
        x = self.linear_out(x)
        x = self.tanh(x)
        imgs = x.view(x.size(0), *self.img_shape)
        
        return imgs
    
    def __str__(self):
        return "CGAN" if self.conditional else "GAN"
    
class Discriminator(nn.Module):
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        conditional: bool = False,
        num_classes: int = 10,
        emb_size: int = 32,
        dropout_rate: float = 0.4,
    ):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.conditional = conditional
        self.num_classes = num_classes
        self.linear_1 = nn.Linear(self.img_shape[0] * self.img_shape[1] * self.img_shape[2], 784)
        if conditional:
            self.embedding = nn.Embedding(num_classes, emb_size)
            self.linear_1 = nn.Linear(self.img_shape[0] * self.img_shape[1] * self.img_shape[2] + emb_size, 784)

        self.activation = nn.LeakyReLU(0.1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_2 = nn.Linear(784, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
    
    def forward(self, imgs: torch.tensor, cond_class: Optional[Union[int, torch.tensor]] = None) -> torch.tensor:
        x = self.flatten(imgs)
        if cond_class is not None:
            if isinstance(cond_class, int):
                cond_class = torch.tensor([cond_class, int], dtype = torch.int32)
            input_embs = self.embedding(cond_class.to(config.DEVICE))
            x = torch.cat((x, input_embs), dim = 1)
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.dropout(self.activation(self.linear_2(x)))
        x = self.dropout(self.activation(self.linear_3(x)))
        probs = self.out(x)
        return probs
    
    def __str__(self):
        return "CDiscriminator" if self.conditional else "Discriminator"
        
        
if __name__ == "__main__":
    generator = Generator(hidden_size = 64, img_shape=(3,28,28), conditional=False).to(config.DEVICE)
    sample_zs = torch.randn(size = (5, 64)).to(config.DEVICE)
    output_imgs = generator(sample_zs)
    print(str(generator),output_imgs.size())
    discriminator = Discriminator(img_shape=(3,28,28), conditional=False).to(config.DEVICE)
    probs = discriminator(output_imgs)
    print(str(discriminator),probs.size())