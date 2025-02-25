# import numpy as np
# import rasterio
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# from PIL import Image

# # ✅ Load Trained Model
# model = load_model("deforestation_model.h5")

# # ✅ Define the Paths for First Tile
# tile_paths = {
#     "2015": r"C:\Users\Asus\Desktop\arf satillite-details\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif",
#     "2024": r"C:\Users\Asus\Desktop\arf satillite-details\2024\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"
# }

# # ✅ Image Size for CNN Model
# IMG_SIZE = (256, 256)
# THRESHOLD = 0.5  # Probability > 0.5 means deforested

# # ✅ Function to Preprocess a Single Tile
# def preprocess_image(tile_path):
#     with rasterio.open(tile_path) as src:
#         image_array = src.read(1)  # Read first band (grayscale)
#         img = Image.fromarray(image_array)

#         # Resize and Normalize
#         img = img.resize(IMG_SIZE)
#         img_array = img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         return img_array

# # ✅ Analyze 2015 and 2024 First Tile
# for year, path in tile_paths.items():
#     img_array = preprocess_image(path)
    
#     # Predict deforestation
#     prob = model.predict(img_array)[0][0]
#     prediction = "Deforested" if prob > THRESHOLD else "Non-Deforested"

#     print(f"📌 {year} - Tile 00000.00000: {prediction} (Score: {prob:.4f})")
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff
import cv2

# Load trained model
MODEL_PATH = "deforestation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess image


def preprocess_image(image_path, target_size=(256, 256)):
    img = tiff.imread(image_path)  # Read TIFF with correct channels
    img = cv2.resize(img, target_size)  # Resize
    img = img.astype(np.float32) / 255.0  # Normalize
    if len(img.shape) == 3:  
        img = img[:, :, :3]  # Keep only the first 3 channels if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Paths to 2015 and 2024 images (example for tile)
tile_id = "Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000"
image_2015_path = rf"C:\Users\Asus\Desktop\arf satillite-details\2015\{tile_id}.tif"
image_2024_path = rf"C:\Users\Asus\Desktop\arf satillite-details\2024\{tile_id}.tif"

# Predict deforestation scores
score_2015 = model.predict(preprocess_image(image_2015_path))[0][0]
score_2024 = model.predict(preprocess_image(image_2024_path))[0][0]

# Convert scores to percentages
score_2015_percentage = score_2015 * 100
score_2024_percentage = score_2024 * 100

# 📊 1. Bar Chart Visualization
plt.figure(figsize=(6, 4))
plt.bar(["2015", "2024"], [score_2015_percentage, score_2024_percentage], color=["blue", "red"])
plt.ylim(0, 100)
plt.ylabel("Deforestation Probability (%)")
plt.title(f"Deforestation Analysis for Tile {tile_id}")
plt.text(0, score_2015_percentage + 2, f"{score_2015_percentage:.2f}%", ha="center")
plt.text(1, score_2024_percentage + 2, f"{score_2024_percentage:.2f}%", ha="center")
plt.show()

# 📈 2. Heatmap (Example for multiple tiles)
tiles = ["Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000"]
scores_2015 = []
scores_2024 = []

for tile in tiles:
    img_2015 = preprocess_image(rf"C:\Users\Asus\Desktop\arf satillite-details\2015\{tile}.tif")
    img_2024 = preprocess_image(rf"C:\Users\Asus\Desktop\arf satillite-details\2024\{tile}.tif")
    scores_2015.append(model.predict(img_2015)[0][0])
    scores_2024.append(model.predict(img_2024)[0][0])

# Create heatmap data
heatmap_data = np.array([scores_2015, scores_2024]) * 100  # Convert to percentages

plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=tiles, yticklabels=["2015", "2024"])
plt.title("Deforestation Heatmap")
plt.xlabel("Tile ID")
plt.ylabel("Year")
plt.show()
