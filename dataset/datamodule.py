from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex

from sklearn.model_selection import train_test_split

from dataset.dataset import SegmentationDataset
from models.deeplabv3 import build_deeplabv3plus
from utils.albumentation import augment, transform
from utils.losses import FocalDiceLoss

from typing import Any


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        data_dir: str | Path,
        num_classes: int,
        in_channels: int,
        ignore_index: int = 0,
        augment: Any | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        learning_rate: float = 1e-3,
        ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
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
            # Create dataset instance
            full_dataset = SegmentationDataset(
                images_dir = self.data_dir / "flair_1_toy_aerial_train",
                masks_dir = self.data_dir / "flair_1_toy_labels_train",
                augment = self.augment,
                transform = transform
            )
            print(f"Lenght full dataset: {len(full_dataset)}")
            
            # Split dataset
            self.train_dataset, self.val_dataset = random_split(
                dataset = full_dataset,
                lengths = [0.8, 0.2],
                generator = torch.Generator().manual_seed(42)
            )
            print(f"Lenght training dataset: {len(self.train_dataset)}")
            print(f"Lenght validation dataset: {len(self.val_dataset)}")
            
            # Set augment for val datasets to None
            self.val_dataset.augment = None
            
            
        if stage in ("test", None):
            self.test_dataset = SegmentationDataset(
                images_dir = self.data_dir / "flair_1_toy_aerial_test",
                masks_dir = self.data_dir / "flair_1_toy_labels_test",
                augment = None,
                transform=transform
            )
            print(f"Lenght test dataset: {len(self.test_dataset)}")
            
            
    # Define dataloaders
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
            
        )