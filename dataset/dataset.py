from pathlib import Path
import numpy as np
import rasterio

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import List, Tuple


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: list[str | Path],
        masks: list[str | Path],
        augment = None,    
        ) -> None:
        
        
        self.images = images
        self.masks = masks
        self.augment = augment
        
        
    def __len__(self) -> int:
        return(len(self.images))
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # cv2 works in BGR -> we need RGB
        with rasterio.open(self.images[idx]) as src:
            image = src.read()
        
        # Min-Max-Scaling from uint8 0-255 -> 0-1
        # image = image.transpose(2, 0, 1)
        image = image / 255.0
            
        
        with rasterio.open(self.masks[idx]) as src:
            mask = src.read(1)
            
        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
 
        