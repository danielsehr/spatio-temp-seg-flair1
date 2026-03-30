from tqdm import tqdm
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map

root = Path("C:/Users/serr_da/Documents/Datasets/flair_1_toy_dataset")

image_input_dir = root / "flair_1_toy_aerial_train"
image_output_dir = root / "flair_1_toy_aerial_train_NDVI"
image_output_dir.mkdir(exist_ok=True)

images = [p for p in image_input_dir.rglob("*.tif")]
# images = images[:20]
print(len(images))


# === CONFIGURATION ===
band_order = [0, 1, 2, 3, 5]  # [Red, Green, Blue, NIR, DGM]

def get_full_stem(path):
    parent = path.parent.stem
    stem = path.stem
    final_stem = f"{parent}/{stem}"
    
    return final_stem


# === NDVI Function ===
def add_ndvi(
    path_str: str| Path,
    red_band: int = 0,
    nir_band: int = 3,
    ) -> None:
    """
    Process a single image: read, compute NDVI, write new GeoTIFF.
    Must be at module level for multiprocessing.
    """
    path = Path(path_str)
    
    try:
        with rasterio.open(path) as src:
            # Read all bands
            image = src.read()  # (C, H, W)
            image = image[:5, ...]
            nodata = src.nodata

            # Extract Red and NIR
            red = image[red_band].astype(np.float32)
            nir = image[nir_band].astype(np.float32)

            # Handle nodata values by creating bool mask
            if nodata is not None:
                mask = (red == nodata) | (nir == nodata)
            else: 
                mask = np.zeros(red.shape, dtype=bool)

        
            # Create ndvi mask
            ndvi = np.full(red.shape, 0, dtype=np.float32)

            # Add small epsilon to avoid 0 division
            denominator = nir + red + 1e-8

            # Bool mask for valid pixels (not nodata and not zero)
            valid = (~mask) & (denominator != 0)

            # Calc NDVI
            ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]


            # Stack NDVI as 5th band
            new_image = np.vstack([image, ndvi[None, ...]])  # (5, H, W)

            # Update profile
            profile = src.profile.copy()
            profile.update(
                dtype=rasterio.float32,
                count=6,
                nodata=nodata,
                compress='lzw'
            )


            relative = path.relative_to(image_input_dir)
            
            output_path = image_output_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write new GeoTIFF
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(new_image.astype(np.float32))


    except Exception as e:
        print(f"[error] Failed to process {path}: {e}")


for path in tqdm(images):
    add_ndvi(path_str=path)

# results = process_map(
#     process_image,
#     images,
#     max_workers=8,
#     desc="Processing images",
#     total=len(images)
# )