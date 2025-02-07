import os
import numpy as np
import rasterio


output_folder = "processed_output/cloud_masked"
os.makedirs(output_folder, exist_ok=True)

def cloud_mask(image, threshold=0.3):
    """Apply cloud masking by setting high-intensity areas to zero"""
    masked_image = np.where(image > threshold, 0, image)
    return masked_image

def apply_cloud_mask(file_path):
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Read first band
        profile = src.profile  # Get metadata

    masked_image = cloud_mask(image)

    # Save masked image
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(masked_image, 1)

    print(f"Saved Cloud Masked Image: {output_path}")
    return output_path
# import os
# import numpy as np
# import rasterio

# output_folder = "processed_output/cloud_masked"
# os.makedirs(output_folder, exist_ok=True)

# def cloud_mask(image, threshold=0.2):
#     """Apply cloud masking by setting high-intensity areas to zero."""
#     # Dynamic thresholding based on image statistics
#     mean_intensity = np.mean(image)
#     std_intensity = np.std(image)
#     dynamic_threshold = mean_intensity + 2 * std_intensity  # Adjust multiplier as needed

#     masked_image = np.where(image > dynamic_threshold, 0, image)  # Mask clouds
#     return masked_image

# def apply_cloud_mask(file_path):
#     """Apply cloud masking to an image."""
#     with rasterio.open(file_path) as src:
#         image = src.read(1).astype(np.float32)  # Read first band
#         profile = src.profile  # Get metadata

#     # Normalize image before masking
#     normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

#     # Apply cloud mask
#     masked_image = cloud_mask(normalized_image)

#     # Save masked image
#     output_path = os.path.join(output_folder, os.path.basename(file_path))
#     profile.update(dtype=rasterio.float32)

#     with rasterio.open(output_path, 'w', **profile) as dst:
#         dst.write(masked_image, 1)

#     print(f"Saved Cloud Masked Image: {output_path}")
#     return output_path
