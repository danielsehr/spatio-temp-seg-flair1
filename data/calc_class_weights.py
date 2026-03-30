import os
from pathlib import Path
import numpy as np
from pandas import read_parquet
import torch
import rasterio
from tqdm import tqdm
import yaml


# Define dataset dir
root = Path("C:/Users/serr_da/Documents/Datasets/flair_1_toy_dataset")
train_masks = root / "flair_1_toy_labels_train"

# Path only
train_data = [p for p in train_masks.rglob("*.tif")]
np_train = np.array(train_data)
print(np_train.shape)


def compute_class_weights(
    mask_paths: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None
    ) -> torch.tensor:

    counts = torch.zeros(num_classes, dtype=torch.long)

    for path in tqdm(mask_paths):
        with rasterio.open(path) as src:
            mask = src.read(1)
        
        mask = torch.from_numpy(mask).long()
        masks = mask.view(-1)

        if ignore_index is not None:
            masks = masks[masks != ignore_index]

        counts += torch.bincount(
            masks.flatten(),
            minlength=num_classes
            )

    freq = counts.float() / counts.sum()
    nonzero = freq > 0
    
    weights = torch.zeros_like(freq)
    weights[nonzero] = torch.median(freq[nonzero]) / freq[nonzero]

    # Normalize (only over valid weights)
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

    return weights


# Calc weights
weights = compute_class_weights(
    np_train, 
    num_classes=19,
    ignore_index=19
    )


print(f"Weights: {weights}")
print(f"Len weights: {len(weights)}")
print(len(weights))

