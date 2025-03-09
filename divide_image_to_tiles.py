
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os

# Load the dataset
gpkg_path = r"C:\Users\Asus\Downloads\prodes_amazonia_nb.gpkg\prodes_amazonia_nb.gpkg"
brazil_states = gpd.read_file(gpkg_path)

# Filter for Rondônia
rondonia = brazil_states[brazil_states["state"] == "RO"]

# Save the filtered data (optional)
rondonia.to_file("rondonia_prodes.gpkg", driver="GPKG")
print("✅ Filtered data saved as rondonia_prodes.gpkg")

# Define output directory
output_dir = "rondonia_tiles"
os.makedirs(output_dir, exist_ok=True)

# Define raster resolution
pixel_size = 0.0001  # Adjust based on desired resolution

# Get bounds of the shapefile
minx, miny, maxx, maxy = rondonia.total_bounds
width = int((maxx - minx) / pixel_size)
height = int((maxy - miny) / pixel_size)

# Define tile size (in pixels)
tile_size_x = 256  # Adjust for smaller/larger images
tile_size_y = 256

# Create an empty raster array
raster = np.zeros((height, width), dtype=np.uint8)

# Rasterize the vector data with 255 instead of 1
rasterized = rasterize(
    [(geom, 255) for geom in rondonia.geometry],  # Change 1 to 255 for better visibility
    out_shape=(height, width),
    transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height),
    fill=0,
    dtype=np.uint8
)

# Function to save each tile
def save_tile(tile_data, tile_x, tile_y, transform):
    tile_filename = os.path.join(output_dir, f"tile_{tile_x}_{tile_y}.tif")
    with rasterio.open(
        tile_filename, "w", driver="GTiff", height=tile_size_y, width=tile_size_x,
        count=1, dtype=tile_data.dtype, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(tile_data, 1)
    print(f"✅ Saved: {tile_filename}")

# Split raster into smaller tiles
for y in range(0, height, tile_size_y):
    for x in range(0, width, tile_size_x):
        # Extract tile data
        tile = rasterized[y:y+tile_size_y, x:x+tile_size_x]
        
        # Ensure tile is not empty
        if np.any(tile > 0):  # Check for non-zero pixels
            # Calculate transform for this tile
            tile_transform = rasterio.transform.from_origin(
                minx + x * pixel_size, maxy - y * pixel_size,  # Fix: Correct tile positioning
                pixel_size, pixel_size
            )
            save_tile(tile, x, y, tile_transform)

print("✅ All tiles saved successfully!")

