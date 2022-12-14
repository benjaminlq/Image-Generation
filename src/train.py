"""Training Script
"""
import argparse
import config

from config import LOGGER
from dataloaders import dataloaders
from engine import BCE_VAE_loss, MSE_VAE_loss, train
from models import models

def get_argument_parser():
    """Input Parameters for training model
    - No of Epochs
    - Batch Size
    - Learning Rate
    - Model Type
    - Load Existing Checkpoint
    - Early Stopping Setting
    - Patience Epoch Count
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
        "-md",
        "--model",
        help="Type of model instance used for training",
        type=str,
        default="BaseVAE",
    )

    parser.add_argument(
        "-l", "--load", help="Load Model from previous ckpt", action="store_true"
    )

    parser.add_argument(
        "-es",
        "--earlystop",
        help="Whether EarlyStop during model training",
        action="store_true",
    )

    parser.add_argument(
        "-p",
        "--patience",
        help="How Many epoches for Early Stopping",
        type=int,
        default=5,
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
        default="mse",
    )
    
    parser.add_argument(
        "-hd",
        "--hidden",
        help="Length of latent vector",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_argument_parser()
    datamodule, img_size = dataloaders[args.dataset]
    if args.loss.lower() == "mse":
        loss_function = MSE_VAE_loss
        data_manager = datamodule(batch_size=args.batch_size, std_normalize=False)
        model = models[args.model](input_size=img_size, hidden_size=args.hidden, activation="Sigmoid", **config.MODEL_PARAMS[args.model])
    elif args.loss.lower() == "bce":
        loss_function = BCE_VAE_loss
        data_manager = datamodule(batch_size=args.batch_size, std_normalize=False)
        model = models[args.model](input_size=img_size, hidden_size=args.hidden, activation="Sigmoid", **config.MODEL_PARAMS[args.model])
    else:
        raise Exception("Incorrect Settings for Loss Function")

    LOGGER.info(f"Training {str(model)} using {args.loss.lower()} loss function")
    
    train(
        model,
        loss_function,
        data_manager,
        no_epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping=args.earlystop,
        patience=args.patience,
        load=args.load,
        save=True,
    )
    LOGGER.info(f"{str(model)}: Training completed")
    
# python3 src/train.py -e 20 -md ConvVAE -ls mse
