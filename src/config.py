"""Config files
"""
import logging
import os
import sys
from pathlib import Path

import torch

MAIN_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = MAIN_PATH / "data"
ARTIFACT_PATH = MAIN_PATH / "artifacts"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_WORKERS = os.cpu_count()

EPOCHS = 20
LEARNING_RATE = 1e-4
EPS = 1e-8
BATCH_SIZE = 32
CLIP = 1.5
ALPHA = 1000.0

### Model Params
MODEL_PARAMS = {
    "BaseVAE": {},
    "DeepVAE": {},
    "ConvVAE": {"kernel_size": 3},
    "GAN": {},
    "Cond_GAN": {},
}

### Logging configurations
LOGGER = logging.getLogger(__name__)

stream_handler = logging.StreamHandler(sys.stdout)
if not (ARTIFACT_PATH / "model_ckpt").exists():
    (ARTIFACT_PATH / "model_ckpt").mkdir()

file_handler = logging.FileHandler(
    filename=str(ARTIFACT_PATH / "model_ckpt" / "logfile.log")
)

formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)
LOGGER.addHandler(file_handler)
