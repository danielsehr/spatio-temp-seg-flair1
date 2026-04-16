from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from dataset.dataset_meta_concat import SegmentationDatasetMetaConcat
from dataset.datamodule_meta_concat import DeepLabV3PlusMetaConcat
from utils.albumentation import augment, transform
from utils.plotting import plot_image_mask


eval_root = Path("C:/Users/Administrator/PythonProjects/landcover_classification/FLAIR1/results")
data_root = Path("C:/Users/Administrator/PythonProjects/landcover_classification/ML_datasets/FLAIR1/flair_1_toy_dataset")
subfolder = "baseline"

# Instantiate Module class
model = DeepLabV3PlusMetaConcat(
    data_dir=data_root,
    num_classes=19, 
    in_channels=5, 
    meta_dim=25, 
    augment=augment,
    transform=transform,
    batch_size=2,
    num_workers=2,
    ignore_index=0,
    learning_rate=1e-03
)

class_weights = torch.tensor([
    0.0000e+00, 2.8986e-03, 4.7309e-03, 1.8890e-03, 7.4763e-03, 6.0552e-03,
    7.3642e-03, 2.2929e-03, 4.6302e-03, 2.3903e-02, 1.7715e-03, 3.3325e-03,
    7.8365e-03, 6.1710e-01, 6.9375e-01, 4.3067e-01, 0.0000e+00, 8.8827e+00,
    6.3016e+00
])


# Checkpoints
checkpoint_dir = eval_root / subfolder / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True, parents=True)


# # Callbacks
# logger_dir = eval_root / subfolder / "logger"
# logger_dir.mkdir(parents=True, exist_ok=True)

# csv_logger = CSVLogger(
#     save_dir=logger_dir,
#     name="csv_logs"
# )

# tb_logger = TensorBoardLogger(
#     logger_dir, 
#     name="tb_logs"
# )

# checkpoint = ModelCheckpoint(
#     monitor="val_loss",
#     dirpath=checkpoint_dir,
#     filename="ckpt-{epoch:02d}-{val_loss:.3f}-{val_iou:.3f}",
#     save_top_k=2,
#     mode="min",
#     save_weights_only=True
# )

# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     min_delta=0.00,
#     patience=20,
#     mode="max",
# )

# lr_monitor = LearningRateMonitor(logging_interval="epoch")

# callbacks = [
#     checkpoint,
#     early_stopping,
#     lr_monitor
# ]


# Define trainer
trainer = Trainer(
    # accelerator="gpu",
    accelerator="cpu",
    max_epochs=200,
    # devices=[1],
    devices="auto",
    enable_progress_bar=True,
    # strategy=DDPStrategy(find_unused_parameters=True),
    # callbacks=callbacks,
    # logger=[csv_logger, tb_logger],
    log_every_n_steps=10
)


# Train model
trainer.fit(model)
torch.nn.Conv2d()