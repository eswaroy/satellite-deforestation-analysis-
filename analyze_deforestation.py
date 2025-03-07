import numpy as np
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Load trained model
MODEL_PATH = "deforestation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to compute NDVI, deforested area, and model probability
def analyze_deforestation(image_path, ndvi_threshold=0.3, target_size=(256, 256)):
    with rasterio.open(image_path) as src:
        red = src.read(4).astype(float)  # Band 4: Red
        nir = src.read(5).astype(float)  # Band 5: NIR
        ndvi = (nir - red) / (nir + red + 1e-10)  # NDVI calculation
        
        # Calculate deforested area percentage (NDVI < threshold)
        deforested_mask = ndvi < ndvi_threshold
        deforested_percentage = (np.sum(deforested_mask) / deforested_mask.size) * 100
        
        # For model prediction: create binary mask and preprocess
        mask = np.where(ndvi < ndvi_threshold, 255, 0).astype(np.uint8)
        mask_resized = cv2.resize(mask, target_size)
        img = mask_resized / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)  # Convert to 3-channel for model
        img = np.expand_dims(img, axis=0)
        model_prob = model.predict(img, verbose=0)[0][0] * 100  # Silent prediction
        
    return ndvi, deforested_percentage, model_prob

# Paths to 2015 and 2024 images
tile_id = "Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000"#"Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000025088"
image_2015_path = f"C:\\Users\\Asus\\Desktop\\arf satillite-details\\2015\\{tile_id}.tif"
image_2024_path = f"C:\\Users\\Asus\\Desktop\\arf satillite-details\\2024\\{tile_id}.tif"

# Analyze deforestation for both years
ndvi_2015, deforested_2015, prob_2015 = analyze_deforestation(image_2015_path, ndvi_threshold=0.3)
ndvi_2024, deforested_2024, prob_2024 = analyze_deforestation(image_2024_path, ndvi_threshold=0.3)

# Calculate change
deforestation_change = deforested_2024 - deforested_2015

# Print results
print(f"2015 Deforested Area: {deforested_2015:.2f}% (Model Probability: {prob_2015:.2f}%)")
print(f"2024 Deforested Area: {deforested_2024:.2f}% (Model Probability: {prob_2024:.2f}%)")
print(f"Deforestation Change (2024 - 2015): {deforestation_change:.2f}%")

# Visualization 1: Bar chart for deforested area
plt.figure(figsize=(6, 4))
plt.bar(["2015", "2024"], [deforested_2015, deforested_2024], color=["blue", "red"])
plt.ylim(0, max(deforested_2015, deforested_2024) + 10)
plt.ylabel("Deforested Area (%)")
plt.title(f"Deforestation Change for Tile {tile_id}")
plt.text(0, deforested_2015 + 2, f"{deforested_2015:.2f}%", ha="center")
plt.text(1, deforested_2024 + 2, f"{deforested_2024:.2f}%", ha="center")
plt.show()

# Visualization 2: NDVI images
plt.figure(figsize=(12, 5))

# 2015 NDVI
plt.subplot(1, 2, 1)
plt.imshow(ndvi_2015, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='NDVI')
plt.title("2015 NDVI")
plt.axis('off')

# 2024 NDVI
plt.subplot(1, 2, 2)
plt.imshow(ndvi_2024, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='NDVI')
plt.title("2024 NDVI")
plt.axis('off')

plt.tight_layout()
plt.show()

# Visualization 3: NDVI difference map
difference = ndvi_2024 - ndvi_2015
plt.figure(figsize=(6, 5))
plt.imshow(difference, cmap='RdBu', vmin=-0.5, vmax=0.5)
plt.colorbar(label='NDVI Change (2024 - 2015)')
plt.title("NDVI Difference")
plt.axis('off')
plt.show()