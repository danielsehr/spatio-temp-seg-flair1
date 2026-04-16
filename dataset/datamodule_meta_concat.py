from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex

from sklearn.model_selection import train_test_split

from dataset.dataset_meta_concat import SegmentationDatasetMetaConcat 
from models.deeplabv3 import build_deeplabv3plus
from utils.albumentation import augment, transform
from utils.losses import FocalDiceLoss

from typing import Any


class DeepLabV3PlusMetaConcat(pl.LightningModule):
    def __init__(
        self,
        data_dir: str | Path,
        num_classes: int,
        in_channels: int,
        meta_dim: int,
        augment: Any,
        transform: Any,
        batch_size: int,
        num_workers: int,
        ignore_index: int = 0,
        learning_rate:float = 1e-03
        ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.meta_dim = meta_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.augment = augment
        self.transform = transform
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.save_hyperparameters()        
        
        
        # Model
        self.model = build_deeplabv3plus(
            num_classes=self.num_classes,
            in_channels=self.in_channels
        )
        
        # Add dimension to conv2d layer in new segmentation head
        seg_head_in_channels = int(self.model.segmentation_head[0].in_channels)
        self.new_in_channels = seg_head_in_channels + self.meta_dim
        
        self.model.segmentation_head[0] = nn.Conv2d(
            in_channels=self.new_in_channels, 
            out_channels=self.num_classes, 
            kernel_size=1
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
            self.full_dataset = SegmentationDatasetMetaConcat(
                images_dir = self.data_dir / "flair_1_toy_aerial_train",
                masks_dir = self.data_dir / "flair_1_toy_labels_train_remap",
                meta_json_dir = self.data_dir / "flair_1_metadata_aerial", 
                augment = self.augment,
                transform = self.transform,
            )
            print(f"Lenght full dataset: {len(self.full_dataset)}")
            
            # Split dataset
            self.train_dataset, self.val_dataset = random_split(
                dataset = self.full_dataset,
                lengths = [0.8, 0.2],
                generator = torch.Generator().manual_seed(42)
            )
            print(f"Lenght training dataset: {len(self.train_dataset)}")
            print(f"Lenght validation dataset: {len(self.val_dataset)}")
            
            # Set augment for val datasets to None
            self.val_dataset.augment = None
            
            
        if stage in ("test", None):
            self.test_dataset = SegmentationDatasetMetaConcat(
                images_dir = self.data_dir / "flair_1_toy_aerial_test",
                masks_dir = self.data_dir / "flair_1_toy_labels_test_remap",
                meta_json_dir = self.data_dir / "flair_1_metadata_aerial", 
                augment = None,
                transform = self.transform,
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
            # persistent_workers=True,
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
        
    
    def forward(self, x, meta):
        # x: (B, C, H, W)
        # meta: (B, meta_dim)
        
        # Encoder
        features = self.model.encoder(x)
        
        # Decoder
        decoder_output = self.model.decoder(features)
        
        B, C, H, W = decoder_output.shape
        
        # Expand meta to match spatial dimensions of x
        meta_expanded = meta.unsqueeze(-1).unsqueeze(-1)  # (B, meta_dim, 1, 1)
        meta_expanded = meta_expanded.expand(-1, -1, H, W)  # (B, meta_dim, H, W)
        
        # Concatenate along channel dimension
        x_concat = torch.cat([decoder_output, meta_expanded], dim=1)  # (B, C + meta_dim, H, W)
        
        # Segmentation head
        return self.model.segmentation_head(x_concat)
        
        
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        bin_gh = batch["binary_geohash"]
        masks = batch["mask"]
        
        print("train step:")
        print("bin_gh:", bin_gh)

        logits = self(images, bin_gh)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)
        self.train_iou(preds, masks)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        bin_gh = batch["binary_geohash"]
        masks = batch["mask"]
        
        logits = self(images, bin_gh)
        loss = self.criterion(logits, masks)
        
        preds = torch.argmax(logits, dim=1)
        self.val_iou(preds, masks)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)        

        return loss
    
    def test_step(self, batch, batch_idx):
        images = batch["image"]
        bin_gh = batch["binary_geohash"]
        masks = batch["mask"]
        
        logits = self(images, bin_gh)
        loss = self.criterion(logits, masks)
        
        preds = torch.argmax(logits, dim=1)
        self.test_iou(preds, masks)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_iou", self.test_iou, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }


    def on_train_epoch_end(self):
        opt = self.optimizers()
        self.log("lr", opt.param_groups[0]["lr"], prog_bar=True, sync_dist=True)


    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        bin_gh = batch["binary_geohash"]
        
        logits = self(images, bin_gh)
        
        return torch.argmax(logits, dim=1)