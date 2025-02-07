import os
import rasterio
import numpy as np

output_folder = "processed_output/ndvi"
os.makedirs(output_folder, exist_ok=True)

def compute_ndvi(nir_path, red_path):
    """Calculate NDVI using NIR and Red bands"""
    with rasterio.open(nir_path) as nir_src:
        nir_band = nir_src.read(1).astype(np.float32)
        profile = nir_src.profile  # Get metadata

    with rasterio.open(red_path) as red_src:
        red_band = red_src.read(1).astype(np.float32)

    # Avoid division by zero
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)

    # Save NDVI image
    output_path = os.path.join(output_folder, "ndvi_result.tif")
    profile.update(dtype=rasterio.float32)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi, 1)

    print(f"Saved NDVI Image: {output_path}")
    return output_path
# import os
# import rasterio
# import numpy as np

# output_folder = "processed_output/ndvi"
# os.makedirs(output_folder, exist_ok=True)

# def compute_ndvi(nir_path, red_path):
#     """Calculate NDVI using NIR and Red bands."""
#     with rasterio.open(nir_path) as nir_src:
#         nir_band = nir_src.read(1).astype(np.float32)
#         profile = nir_src.profile  # Get metadata

#     with rasterio.open(red_path) as red_src:
#         red_band = red_src.read(1).astype(np.float32)

#     # Avoid division by zero
#     ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)

#     # Save NDVI image
#     output_path = os.path.join(output_folder, "ndvi_result.tif")
#     profile.update(dtype=rasterio.float32)

#     with rasterio.open(output_path, 'w', **profile) as dst:
#         dst.write(ndvi, 1)

#     print(f"Saved NDVI Image: {output_path}")
#     return output_path


