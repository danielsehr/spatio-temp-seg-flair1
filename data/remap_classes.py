import rasterio
from pathlib import Path
import numpy as np

# Config
data_root = Path("C:/Users/serr_da/Documents/Datasets/flair_1_toy_dataset")
masks_dir = data_root / "flair_1_toy_labels_test"      # folder with original masks
output_dir = data_root / Path("flair_1_toy_labels_test_remap")     # folder to save remapped masks

remap_dict = {19: 0}  # remap class 19 -> 0

# Remap
for mask_path in masks_dir.rglob("*.tif"):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # read first band
        profile = src.profile.copy()

    # Remap values
    for old_val, new_val in remap_dict.items():
        mask[mask == old_val] = new_val

    # Save remapped mask
    relative = mask_path.relative_to(masks_dir)

    out_path = output_dir / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(out_path)

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask, 1)

    print(f"Processed {mask_path.name}")