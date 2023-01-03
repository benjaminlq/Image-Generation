"""Unit Tests for models
"""
import torch

import config
from models import *
from models import models


def test_vae_models():
    """Unit Tests for VAE models"""
    for modelname, model_type in models.items():
        if "VAE" in modelname:
            sample_batch = torch.rand(5, 3, 224, 224)
            model = model_type(input_size=(3, 224, 224), hidden_size=20)
            if hasattr(model, "num_classes"):
                labels = torch.randint(0, 9, (5,))
                out, mu, log_var = model(sample_batch, labels)
            else:
                out, mu, log_var = model(sample_batch)
            assert out.size() == torch.Size(
                (5, 3, 224, 224)
            ), f"{modelname}: Wrong Reconstruction Image Dimension"
            assert mu.size() == torch.Size(
                (5, 20)
            ), f"{modelname}: Wrong Latent Distribution Mean Dimension"
            assert log_var.size() == torch.Size(
                (5, 20)
            ), f"{modelname}: Wrong Latent Distribution Variance Dimension"


def test_gan_models():
    """Unit Tests for GAN models"""
    img_shape = (3, 28, 28)
    sample_zs = torch.randn(size=(5, 64)).to(config.DEVICE)
    generator = Generator(input_size=img_shape, hidden_size=64)
    discriminator = Discriminator(input_size=img_shape)
    generator = generator.to(config.DEVICE)
    discriminator = discriminator.to(config.DEVICE)
    gen_imgs = generator(sample_zs)
    assert gen_imgs.size() == torch.Size(
        (5, 3, 28, 28)
    ), "Wrong Dimension for Generated Images"
    probs = discriminator(gen_imgs)
    assert probs.size() == torch.Size((5, 1)), "Wrong Discriminator output"

    generator = CGenerator(input_size=img_shape, hidden_size=64)
    discriminator = CDiscriminator(input_size=img_shape)
    cond_class = torch.tensor([1, 2, 4, 5, 6], dtype=torch.int32, device=config.DEVICE)
    generator = generator.to(config.DEVICE)
    discriminator = discriminator.to(config.DEVICE)
    gen_imgs = generator(sample_zs, cond_class)
    assert gen_imgs.size() == torch.Size(
        (5, 3, 28, 28)
    ), "Wrong Dimension for Generated Images using Conditional Generator"
    probs = discriminator(gen_imgs, cond_class)
    assert probs.size() == torch.Size((5, 1)), "Wrong Conditional Discriminator output"
