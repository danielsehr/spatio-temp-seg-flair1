import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex

from sklearn.model_selection import train_test_split

from models.deeplabv3 import build_deeplabv3plus
from utils.albumentation import augment
from utils.losses import FocalDiceLoss

from typing import Any


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        images, 
        masks,
        num_classes: int,
        in_channels: int,
        ignore_index: int = 0,
        augment: Any | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        learning_rate: float = 1e-3,
        ):
        super().__init__()
        
        self.images = images
        self.masks = masks
        
        self.ignore_index = ignore_index
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        
        self.save_hyperparameters(ignore=["images", "masks"])    
        
        
        # Model
        self.model = build_deeplabv3plus(
            num_classes=self.num_classes,
            in_channels=self.in_channels
        )
        
        
        # Loss & Metrics
        self.criterion = FocalDiceLoss(
            alpha=0.25,
            gamma=2.0,
            smooth=1.0,
            dice_weight=1.0,
            focal_weight=1.0,
            ignore_index=self.ignore_index
        )
        
        
        # Metrics
        self.train_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        self.test_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        
    
    def setup(self, stage=None):
        if stage in ("fit", None):
            X, Y = random_split(
                dataset=self.images,
                lengths=[0.8, 0.2],
                generator=torch.Generator.manual_seed(42)
            )
            