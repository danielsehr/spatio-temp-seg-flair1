from typing import Optional
import json

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import contextily as ctx

from pyproj import Transformer
import pygeohash as pgh
from pygeohash.viz import plot_geohash

import rasterio
import geopandas as gpd
import shapely
from shapely.geometry import Point

from dataset.dataset_meta_concat import SegmentationDatasetMetaConcat
from utils.geohash import add_lon_lat, add_geohash_to_metadata, create_geohash_bbox


#--- File paths ---
dataset_dir = Path("C:/Users/Administrator/PythonProjects/landcover_classification/ML_datasets/FLAIR1/")
json_path = dataset_dir / "flair_1_metadata_aerial"
jsons = [p for p in json_path.rglob("*.json")]




#--- Create geohashes in metadata ---
json_path = [p for p in jsons if "_geohash" not in str(p)][0]

with open(json_path) as f:
    metadata = json.load(f)
    
    
# Add lon lat from centroid
new_metadata = add_lon_lat(metadata=metadata)

# Create geohash and add to dict
add_geohash_to_metadata(metadata=new_metadata)

# Save dict to json for ML
save_path = json_path / f"{json_path.stem}_geohash.json"

with open(save_path, "w") as f:
    json.dump(new_metadata, f, indent=2)
    


#--- Geodataframe Bboxes ---
# Create geodataframe with geohash bounding boxes and save to disk
save_dir = dataset_dir / "flair_1_toy_dataset" / "shapefiles"

gdf_gh_bbox = create_geohash_bbox(metadata=new_metadata, save_dir=save_dir)



#--- Plot geohash bboxes ---
# Convert to Web Mercator for contextily
gdf_gh_bbox_3875 = gdf_gh_bbox.to_crs(epsg=3857)

ax = gdf_gh_bbox_3875.plot(figsize=(8, 8), color="red", markersize=50)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
ax.set_axis_off()
plt.show()