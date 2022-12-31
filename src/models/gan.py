"""GAN Model
"""
import torch
import torch.nn as nn

from typing import Optional, Union, Literal
from models.block import MLPBlock
import config

class Generator(nn.Module):
    """Generator Module
    """
    def __init__(
        self,
        img_shape: tuple = (3, 28, 28),
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
        if self.conditional:
            self.cond_size = emb_size
            self.hidden_size += emb_size
            self.num_classes = num_classes
            self.embedding = nn.Embedding(num_classes, emb_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.linear1 = MLPBlock(self.hidden_size, 196)
        self.linear2 = MLPBlock(196, 392)
        self.linear3 = MLPBlock(392, 784)
        self.linear4 = MLPBlock(784, self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        self.tanh = nn.Tanh()
        
    def forward(self, z: torch.tensor, cond_class: Optional[Union[int, torch.tensor]] = None) -> torch.tensor:
        """Forward Pass

        Args:
            z (torch.tensor): Noise Latent vectors to feed to Generator
            cond_class (Optional[Union[int, list]], optional): Class or List of class for conditional GAN. Model P(z|c). Defaults to None.

        Returns:
            torch.tensor: Generated Images
        """
        if cond_class:
            if isinstance(cond_class, int):
                cond_class = torch.tensor([cond_class], dtype = torch.int32)
            input_embs = self.embedding(cond_class.to(config.DEVICE))
            z = torch.cat((z, input_embs), dim = 1)
        
        x = self.dropout(self.linear1(z))
        x = self.dropout(self.linear2(x))
        x = self.dropout(self.linear3(x))
        x = self.dropout(self.linear4(x))
        x = self.tanh(x)
        imgs = x.view(x.size(0), *self.img_shape)
        
        return imgs
    
class Discriminator(nn.Module):
    def __init__(
        self,
        img_shape: tuple = (3, 28, 28),
        conditional: bool = False,
        num_classes: int = 10,
        emb_size: int = 32,
        dropout_rate: float = 0.3,
    ):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.conditional = conditional
        self.linear1 = MLPBlock(self.img_shape[0] * self.img_shape[1]*self.img_shape[2], 784)
        if conditional:
            self.embedding = nn.Embedding(num_classes, emb_size)
            self.num_classes = num_classes
            self.linear1 = MLPBlock(self.img_shape[0] * self.img_shape[1]*self.img_shape[2] + emb_size, 784)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear2 = MLPBlock(784, 392)
        self.linear3 = MLPBlock(392, 196)
        self.linear4 = MLPBlock(196, 64)
        self.out = nn.Linear(64, 1)
    
    def forward(self, imgs: torch.tensor, cond_class: Optional[Union[int, torch.tensor]] = None) -> torch.tensor:
        x = self.flatten(imgs)
        if cond_class:
            if isinstance(cond_class, int):
                cond_class = torch.tensor([cond_class, int], dtype = torch.int32)
            input_embs = self.embedding(cond_class.to(config.DEVICE))
            x = torch.cat((x, input_embs), dim = 1)
        x = self.dropout(self.linear1(x))
        x = self.dropout(self.linear2(x))
        x = self.dropout(self.linear3(x))
        x = self.dropout(self.linear4(x))
        probs = self.out(x)
        return probs
        
class GAN:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        img_shape: tuple = (3, 28, 28),
        hidden_size: int = 64,
        conditional: bool = False,
        num_classes: int = 10,
        emb_size: int = 32,
        learning_rate: float = 1e-3,
        k: int = 1,
        loss: Literal["mse","bce"] = "bce",
    #    label_smoothing: float = 0.0,
    ):
        self.img_shape = img_shape
        self.hidden_size = hidden_size
        self.conditional = conditional
        self.emb_size = emb_size
        self.k = k
    #    self.label_smoothing = label_smoothing
        
        if loss == "bce":
            self.adversarial_loss = nn.BCEWithLogitsLoss()
        else:
            self.adversarial_loss = nn.MSELoss()
        
        self.generator = generator(
            img_shape=img_shape,
            hidden_size=hidden_size,
            conditional=conditional,
            emb_size=emb_size,
            num_classes=num_classes)
        self.optimizer_G = torch.optim.Adam(params=self.generator.params(), lr=learning_rate, betas=(0.5, 0.999))
        
        self.discriminator = discriminator(
            img_shape=img_shape,
            conditional=conditional,
            emb_size=emb_size,
            num_classes=num_classes
        )
        self.optimizer_D = torch.optim.Adam(params=self.discriminator.params(), lr=learning_rate, betas=(0.5, 0.999))
        
    def train_discriminator(
        self,
        real_imgs: torch.tensor,
        fake_imgs: torch.tensor,
        real_labels: Optional[Union[int, torch.tensor]] = None,
        fake_labels: Optional[Union[int, torch.tensor]] = None,
        ):
        self.optimizer_D.zero_grad()
        real_bs = real_imgs.size(0)
        real_targets = torch.ones((real_bs, 1), requires_grad=False)
        if self.label_smoothing:
            assert self.label_smoothing >= 0 and self.label_smoothing < 1, "Invalid smoothing factor (must be between 0 and 1)"
            real_targets = real_targets - self.label_smoothing
        real_loss_D = self.adversarial_loss(
            self.discriminator(real_imgs, real_labels), 
        )
        

if __name__ == "__main__":
    generator = Generator(hidden_size = 64, img_shape=(3,28,28), conditional=False).to(config.DEVICE)
    sample_zs = torch.randn(size = (5, 64)).to(config.DEVICE)
    output_imgs = generator(sample_zs)
    print(output_imgs.size())
    discriminator = Discriminator(img_shape=(3,28,28), conditional=False).to(config.DEVICE)
    probs = discriminator(output_imgs)
    print(probs.size())