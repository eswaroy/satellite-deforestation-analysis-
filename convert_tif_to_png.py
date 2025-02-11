# import os
# from PIL import Image
# import rasterio

# # Define input and output directories
# input_dir = "rondonia_tiles"
# output_dir = r"C:\Users\Asus\Desktop\arf satillite-details\tile_png"
# os.makedirs(output_dir, exist_ok=True)

# # Convert all TIFF images to PNG
# for file in os.listdir(input_dir):
#     if file.endswith(".tif"):
#         input_path = os.path.join(input_dir, file)
#         output_path = os.path.join(output_dir, file.replace(".tif", ".png"))

#         with rasterio.open(input_path) as src:
#             image_array = src.read(1)
#             img = Image.fromarray(image_array)
#             img.save(output_path)

# print("✅ All TIFF images converted to PNG!")
import os
import rasterio
import numpy as np
from PIL import Image

# Define input and output directories
input_dir = r"C:\Users\Asus\Desktop\arf satillite-details\rondonia_tiles"
output_deforested = r"C:\Users\Asus\Desktop\arf satillite-details\aa\deforested"
output_non_deforested = r"C:\Users\Asus\Desktop\arf satillite-details\aa\non_deforested"

# Create output directories if they don't exist
os.makedirs(output_deforested, exist_ok=True)
os.makedirs(output_non_deforested, exist_ok=True)

# Classification threshold (change if needed)
THRESHOLD = 0.05  # 5% of pixels as white → deforested

# Process all TIFF files
for file in os.listdir(input_dir):
    if file.endswith(".tif"):
        input_path = os.path.join(input_dir, file)

        # Open the TIFF file
        with rasterio.open(input_path) as src:
            image_array = src.read(1)  # Read first band
            img = Image.fromarray(image_array)

            # Normalize (0 to 1)
            norm_array = image_array / 255.0

            # Calculate the percentage of deforested pixels (white pixels)
            white_pixel_ratio = np.mean(norm_array > 0.5)  # Count white pixels

            # Classify and save the image in the respective folder
            if white_pixel_ratio > THRESHOLD:
                output_path = os.path.join(output_deforested, file.replace(".tif", ".png"))
            else:
                output_path = os.path.join(output_non_deforested, file.replace(".tif", ".png"))

            # Save as PNG
            img.save(output_path)
            print(f"✅ Saved: {output_path} (White Pixels: {white_pixel_ratio:.2%})")

print("✅ All images converted, classified, and saved!")
