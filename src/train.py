"""Training Script
"""
import argparse

import config

# from config import LOGGER
# from models.vae import VAE
# from engine import train
# from dataset import MNIST_Dataloader


def get_argument_parser():
    """Input Parameters for training model
    - No of Epochs
    - Batch Size
    - Learning Rate
    - Model Type
    - Load Existing Checkpoint
    - Early Stopping Setting
    - Patience Epoch Count

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
        help="The number of images in a batch (default: 64)",
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
        "-md",
        "--model",
        help="Type of model instance used for training",
        type=str,
        default="VAE",
    )

    parser.add_argument(
        "-l", "--load", help="Load Model from previous ckpt", type=str, default=False
    )

    parser.add_argument(
        "-es",
        "--earlystop",
        help="Whether EarlyStop during model training",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-p",
        "--patience",
        help="How Many epoches for Early Stopping",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_argument_parser()
