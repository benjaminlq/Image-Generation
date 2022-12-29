import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union

class Generator(nn.Module):
    def __init__(
        self,
        img_size: tuple = (3, 28, 28),
        conditional: bool = False,
        cond_size: int = 32,
        num_classes: int = 10,
    ):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.conditional = conditional
        if conditional:
            self.cond_size = cond_size
            self.num_classes = num_classes
            self.embedding = nn.Embedding(num_classes, cond_size)
            
        self.flatten = nn.Flatten()
        

        
    def forward(self, z: torch.tensor, cond_class: Optional[Union[int, list]] = None):
        if cond_class:
            if isinstance(cond_class, int):
                cond_class = [cond_class]
            input_classes = torch.tensor(cond_class, dtype=torch.int32)
            z = torch.cat((z, input_classes), dim = 1)
        
        
    
class Discriminator(nn.Module):
    