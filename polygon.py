import rasterio
import geopandas as gpd
from rasterio.mask import mask

# Load Brazil states shapefile (PRODES dataset)
brazil_states = gpd.read_file(r"C:\Users\Asus\Downloads\prodes_amazonia_nb.gpkg\prodes_amazonia_nb.gpkg")

# Print unique state names to verify the correct format
print(brazil_states["state"].unique())

# Filter for Rondônia
rondonia = brazil_states[brazil_states["state"] == "Rondônia"]  # Ensure this matches exactly

# Open the PRODES raster
with rasterio.open(r'rondonia_prodes.tif') as src:  # Update path
    # Reproject boundary to match raster CRS
    rondonia = rondonia.to_crs(src.crs)

    # Convert to JSON geometry for masking
    rondonia_geom = [rondonia.geometry.unary_union.__geo_interface__]

    # Crop the raster
    out_image, out_transform = mask(src, rondonia_geom, crop=True)

    # Update metadata
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

# Save cropped image
output_path = r"C:\Users\Asus\Downloads\rondonia_prodes.tif"
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"✅ Cropping complete! New file saved: {output_path}")
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

# Load the dataset
gpkg_path = r"C:\Users\Asus\Downloads\prodes_amazonia_nb.gpkg\prodes_amazonia_nb.gpkg"
brazil_states = gpd.read_file(gpkg_path)

# Filter for Rondônia
rondonia = brazil_states[brazil_states["state"] == "RO"]  # Check "RO" or full name if needed

# Save the filtered data (optional)
rondonia.to_file("rondonia_prodes.gpkg", driver="GPKG")
print("✅ Filtered data saved as rondonia_prodes.gpkg")

# # Define the output raster file
# output_raster = "rondonia_prodes.tif"

# # Define raster resolution (change based on requirements)
# pixel_size = 0.0001  # Adjust based on desired resolution

# # Get bounds of the shapefile
# minx, miny, maxx, maxy = rondonia.total_bounds
# width = int((maxx - minx) / pixel_size)
# height = int((maxy - miny) / pixel_size)

# # Create an empty raster array
# raster = np.zeros((height, width), dtype=np.uint8)

# # Rasterize the vector data
# rasterized = rasterize(
#     [(geom, 1) for geom in rondonia.geometry],
#     out_shape=(height, width),
#     transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height),
#     fill=0,
#     dtype=np.uint8
# )

# # Write the raster file
# with rasterio.open(
#     output_raster, "w", driver="GTiff", height=height, width=width,
#     count=1, dtype=rasterized.dtype, crs="EPSG:4326",
#     transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
# ) as dst:
#     dst.write(rasterized, 1)

# print(f"✅ Raster file saved as {output_raster}")
