"""Training Script
"""
import config
from config import LOGGER
from dataloaders import dataloaders
from engine import BCE_VAE_loss, MSE_VAE_loss, train
from models import models
import utils

if __name__ == "__main__":
    args = utils.get_argument_parser()
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
