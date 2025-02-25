import os
import numpy as np
import rasterio
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Convert TIFF to PNG
input_dir = r"C:\Users\Asus\Desktop\arf satillite-details\rondonia_tiles"       # Input TIFF folder
output_dir = r"C:\Users\Asus\Desktop\arf satillite-details\a"  # Output PNG folder
os.makedirs(output_dir, exist_ok=True)

print("📂 Converting TIFF to PNG...")
for file in os.listdir(input_dir):
    if file.endswith(".tif"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".tif", ".png"))

        with rasterio.open(input_path) as src:
            image_array = src.read(1)
            img = Image.fromarray(image_array)
            img.save(output_path)
print("✅ All TIFF images converted to PNG!")

# Step 2: Normalize function
def load_and_normalize_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Step 3: Augment Data
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

# Step 4: Prepare Training and Validation Dataset
train_generator = datagen.flow_from_directory(
    output_dir,  
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    output_dir,
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode="binary",
    subset="validation"
)

print("✅ Dataset is ready for deep learning model!")

