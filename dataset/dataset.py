from pathlib import Path
import numpy as np
import rasterio

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import List, Tuple, Any


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        augment: Any | None = None,
        transform: Any | None = None, 
        ) -> None:
        
        
        self.images = sorted(Path(images_dir).rglob("*.tif"))
        self.masks = sorted(Path(masks_dir).rglob("*.tif"))
        self.augment = augment
        self.transform = transform
        
        assert len(self.images) == len(self.masks), "Mismatch lenght images/masks"
        
    def __len__(self) -> int:
        return(len(self.images))
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # cv2 works in BGR -> we need RGB
        with rasterio.open(self.images[idx]) as src:
            image = src.read()
        
        with rasterio.open(self.masks[idx]) as src:
            mask = src.read(1)

        # Convert to channel last
        image = image.transpose(1, 2, 0)
        
        # Augmentation
        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert back to channel first
        image = image.transpose(2, 0, 1)
        

        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
 
        