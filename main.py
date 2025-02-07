from preprocess import process_folder
from cloud_masking import apply_cloud_mask
from ndvi_analysis import compute_ndvi

folder_path = r"C:\Users\Asus\Desktop\satillite\2015"

# Step 1: Preprocess images (Normalize & Resize)
preprocessed_images = process_folder(folder_path)

# Step 2: Apply cloud masking
masked_images = {file: apply_cloud_mask(preprocessed_images[file]) for file in preprocessed_images.keys()}

# Step 3: Compute NDVI
nir_path = preprocessed_images[r"C:\Users\Asus\Desktop\satillite\2015\LC08_L2SP_225064_20150816_20200908_02_T1_SR_B4.TIF"]  # NIR Band
red_path = preprocessed_images[r"C:\Users\Asus\Desktop\satillite\2015\LC08_L2SP_225064_20150816_20200908_02_T1_SR_B3.TIF"]  # Red Band
ndvi_result = compute_ndvi(nir_path, red_path)

# print("All processing completed successfully!")
# from preprocess import process_folder
# from cloud_masking import apply_cloud_mask
# from ndvi_analysis import compute_ndvi
# import os

# folder_path = r"C:\Users\Asus\Desktop\satillite\2024"  # Change to your image folder path

# # Step 1: Process raw images (Preprocessing)
# preprocessed_images = process_folder(folder_path)

# # Step 2: Apply cloud masking on the preprocessed images
# preprocessed_folder = "processed_output"
# preprocessed_images = [os.path.join(preprocessed_folder, file) for file in os.listdir(preprocessed_folder) if file.endswith('.TIF')]
# cloud_masked_images = [apply_cloud_mask(image_path) for image_path in preprocessed_images]

# # Step 3: Compute NDVI using the NIR and Red bands (raw images)
# raw_bands_folder = r"C:\Users\Asus\Desktop\satillite\2024\raw_bands"  # Ensure this folder has raw bands (B1-B7)
# raw_images = [os.path.join(raw_bands_folder, file) for file in os.listdir(raw_bands_folder) if file.endswith('.TIF')]
# nir_band_path = raw_images[4]  # NIR Band (usually B5 or similar)
# red_band_path = raw_images[3]  # Red Band (usually B4 or similar)

# ndvi_result = compute_ndvi(nir_band_path, red_band_path)

# print("All processing completed successfully!")




