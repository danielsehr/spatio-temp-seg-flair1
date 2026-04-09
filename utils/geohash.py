from pathlib import Path

import geopandas as gpd
import pygeohash as pgh
from shapely import box
from pyproj import Transformer


def add_lon_lat(metadata: dict) -> dict:
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    for key in metadata:
        item = metadata[key]
        x = item["patch_centroid_x"]
        y = item["patch_centroid_y"]
        lon, lat = transformer.transform(x, y)
        item["lon"] = lon
        item["lat"] = lat

        metadata[key] = item
    
    return metadata


def create_binary_geohash(
    latitude: float,
    longitude: float,
    precision: int = 12
    ):
    
    geohash = pgh.encode(latitude=latitude, longitude=longitude, precision=precision)
    
    # Convert back to the underlying binary string manually
    # Base32 char -> 5 bits based on geohash alphabet
    alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
    bitstr = "".join(f"{alphabet.index(c):05b}" for c in geohash)

    # Use that bitstr as a binary feature vector
    binary_gh = [int(b) for b in bitstr]
    
    return binary_gh, geohash


def add_geohash_to_metadata(metadata: dict) -> None:
    for k, v in metadata.items():
        lat = v["lat"]
        lon = v["lon"]
        
        binary_gh, geohash = create_binary_geohash(latitude=lat, longitude=lon, precision=5)
        
        v["geohash"] = geohash
        v["binary_geohash"] = binary_gh


def create_geohash_bbox(
    metadata: dict,
    save_dir: str | Path | None = None,
    filename: str | None = "geohashes_bbox"
    ) -> gpd.GeoDataFrame:

    # Create polygons in list
    polygons = []

    for k, val in metadata.items():
        geohash = val["geohash"]
        bbox = pgh.get_bounding_box(geohash)
        
        polygons.append(box(
            xmin = bbox.min_lon, 
            ymin = bbox.min_lat,
            xmax = bbox.max_lon,
            ymax = bbox.max_lat,
            ))
        
    # Create gdf containing polygons
    gdf = gpd.GeoDataFrame(
        {
            "image": [f for f in metadata.keys()],
            "geohash": [f["geohash"] for f in metadata.values()]
        },
        geometry=polygons,
        crs="EPSG:4326"
    )
    
    # Optional save as gpkg
    if save_dir:
        save_path = Path(save_dir) / f"{filename}.gpkg" 
        gdf.to_file(save_path, layer="geohashes", driver="GPKG")

    return gdf