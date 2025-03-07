
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
import cv2

# Load trained model
MODEL_PATH = "deforestation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to compute NDVI and create binary mask
def preprocess_landsat_to_mask(image_path, target_size=(256, 256), ndvi_threshold=0.3):
    with rasterio.open(image_path) as src:
        red = src.read(4).astype(float)  # Band 4: Red
        nir = src.read(5).astype(float)  # Band 5: NIR
        ndvi = (nir - red) / (nir + red + 1e-10)  # Avoid division by zero
        # Create binary mask: 255 for deforested (low NDVI), 0 for non-deforested
        mask = np.where(ndvi < ndvi_threshold, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, target_size)
        img = mask / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.repeat(img, 3, axis=-1)  # Convert to 3-channel RGB for model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Paths to 2015 and 2024 images
tile_id = "Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000"#"Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000025088"
image_2015_path = f"C:\\Users\\Asus\\Desktop\\arf satillite-details\\2015\\{tile_id}.tif"
image_2024_path = f"C:\\Users\\Asus\\Desktop\\arf satillite-details\\2024\\{tile_id}.tif"

# Predict deforestation scores
score_2015 = model.predict(preprocess_landsat_to_mask(image_2015_path))[0][0]
score_2024 = model.predict(preprocess_landsat_to_mask(image_2024_path))[0][0]

# Convert scores to percentages
score_2015_percentage = score_2015 * 100
score_2024_percentage = score_2024 * 100

print(f"2015 Deforestation Probability: {score_2015_percentage:.2f}%")
print(f"2024 Deforestation Probability: {score_2024_percentage:.2f}%")

# Bar Chart Visualization
plt.figure(figsize=(6, 4))
plt.bar(["2015", "2024"], [score_2015_percentage, score_2024_percentage], color=["blue", "red"])
plt.ylim(0, 100)
plt.ylabel("Deforestation Probability (%)")
plt.title(f"Deforestation Analysis for Tile {tile_id}")
plt.text(0, score_2015_percentage + 2, f"{score_2015_percentage:.2f}%", ha="center")
plt.text(1, score_2024_percentage + 2, f"{score_2024_percentage:.2f}%", ha="center")
plt.show()