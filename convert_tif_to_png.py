import os
from PIL import Image
import rasterio

# Define input and output directories
input_dir = "rondonia_tiles"
output_dir = r"C:\Users\Asus\Desktop\arf satillite-details\tile_png"
os.makedirs(output_dir, exist_ok=True)

# Convert all TIFF images to PNG
for file in os.listdir(input_dir):
    if file.endswith(".tif"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".tif", ".png"))

        with rasterio.open(input_path) as src:
            image_array = src.read(1)
            img = Image.fromarray(image_array)
            img.save(output_path)

print("✅ All TIFF images converted to PNG!")
