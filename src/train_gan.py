"""Training Script
"""
import argparse
import config

from config import LOGGER
from dataloaders import dataloaders
from engine import train_gan
from models import Generator, Discriminator

def get_argument_parser():
    """Input Parameters for training model
    - No of Epochs
    - Batch Size
    - Learning Rate
    - Model Type
    - Load Existing Checkpoint
    - Save Model
    - Loss Function used
    - Dataset used
    - Latent Hidden Size

    Returns:
        args: Arguments for training models
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        help="How many epochs you need to run (default: 10)",
        type=int,
        default=config.EPOCHS,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help="The number of images in a batch (default: 32)",
        type=int,
        default=config.BATCH_SIZE,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate used for optimizer (default: 1e-4)",
        type=float,
        default=config.LEARNING_RATE,
    )

    parser.add_argument(
        "-c",
        "--conditional",
        help="Type of model instance used for training",
        action="store_true"
    )

    parser.add_argument(
        "-l",
        "--load",
        help="Load Model from previous ckpt",
        action="store_true"
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="Which Dataset to use",
        type=str,
        default="mnist",
    )

    parser.add_argument(
        "-ls",
        "--loss",
        help="Which Loss to use",
        type=str,
        default="bce",
    )
    
    parser.add_argument(
        "-hd",
        "--hidden",
        help="Length of latent vector",
        type=int,
        default=128,
    )
    
    parser.add_argument(
        "-k",
        "--d_g_ratio",
        help="Number of discriminator updates for each generator update",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_argument_parser()
    datamodule, img_size = dataloaders[args.dataset]
    data_manager = datamodule(batch_size = args.batch_size, std_normalize=True)
    generator = Generator(img_shape=img_size, hidden_size=args.hidden, conditional=args.conditional)
    discriminator = Discriminator(img_shape=img_size, conditional=args.conditional)

    LOGGER.info(f"Training {str(generator)} using {args.loss.lower()} loss function")
    
    train_gan(
        generator=generator,
        discriminator=discriminator,
        data_manager=data_manager,
        loss_function=args.loss,
        no_epochs=args.epochs,
        learning_rate=args.learning_rate,
        k=args.d_g_ratio,
        load=args.load,
        save=True,
    )
    LOGGER.info(f"{str(generator)}: Training completed")
    
# python3 src/train_gan.py -e 50 -bs 32 -ls bce -d mnist -c False