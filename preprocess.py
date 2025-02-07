import os
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt

output_folder = "processed_output"
os.makedirs(output_folder, exist_ok=True)

def histogram_stretch(image):
    """Enhance contrast using histogram stretching."""
    p2, p98 = np.percentile(image, (2, 98))
    return np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)

def preprocess_image(file_path, output_size=(256, 256)):
    """Process image: normalize, resize, and save final processed image."""
    with rasterio.open(file_path) as src:
        image = src.read(1).astype(np.float32)  # Read the first band and convert to float32
        profile = src.profile  # Get image metadata

    # Apply contrast enhancement (Histogram Stretching)
    enhanced_image = histogram_stretch(image)

    # Resize the image
    resized_image = cv2.resize(enhanced_image, output_size)

    return resized_image, profile  # Return the final processed image and profile

def save_image(final_image, profile, file_path):
    """Save the final processed image once."""
    # Update the profile with new size
    profile.update({"height": final_image.shape[0], "width": final_image.shape[1], "dtype": "float32"})
    
    # Save final image to disk
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_image, 1)

    print(f"Saved final image: {output_path}")
    return output_path

def process_folder(folder_path):
    """Process all .tif images in the given folder and save final images."""
    tif_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.TIF')]
    processed_images = {}

    # Process each image and save final once
    for file in tif_files:
        print(f"Processing {file}...")

        # Preprocess the image (enhance, resize)
        final_image, profile = preprocess_image(file)

        # Save the final processed image
        output_path = save_image(final_image, profile, file)
        
        processed_images[file] = output_path  # Store the saved image path

    return processed_images

# Run the process on the folder of .tif images
folder_path = r"C:\Users\Asus\Desktop\satillite\2015"
processed_images = process_folder(folder_path)
# import os
# import rasterio
# import numpy as np
# import cv2

# output_folder = "processed_output"
# os.makedirs(output_folder, exist_ok=True)

# def histogram_stretch(image):
#     """Enhance contrast using histogram stretching."""
#     p2, p98 = np.percentile(image, (2, 98))  # Use 2nd and 98th percentiles for robust stretching
#     return np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)

# def normalize_image(image):
#     """Normalize image to [0, 1] range."""
#     return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

# def preprocess_image(file_path, output_size=(256, 256)):
#     """Process image: normalize, resize, and save final processed image."""
#     with rasterio.open(file_path) as src:
#         image = src.read(1).astype(np.float32)  # Read the first band and convert to float32
#         profile = src.profile  # Get image metadata

#     # Normalize and enhance contrast
#     normalized_image = normalize_image(image)
#     enhanced_image = histogram_stretch(normalized_image)

#     # Resize the image
#     resized_image = cv2.resize(enhanced_image, output_size)

#     return resized_image, profile  # Return the final processed image and profile

# def save_image(final_image, profile, file_path):
#     """Save the final processed image once."""
#     # Update the profile with new size
#     profile.update({"height": final_image.shape[0], "width": final_image.shape[1], "dtype": "float32"})
    
#     # Save final image to disk
#     output_path = os.path.join(output_folder, os.path.basename(file_path))
    
#     with rasterio.open(output_path, 'w', **profile) as dst:
#         dst.write(final_image, 1)

#     print(f"Saved final image: {output_path}")
#     return output_path

# def process_folder(folder_path):
#     """Process all .tif images in the given folder and save final images."""
#     tif_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.TIF')]
#     processed_images = {}

#     # Process each image and save final once
#     for file in tif_files:
#         print(f"Processing {file}...")

#         # Preprocess the image (enhance, resize)
#         final_image, profile = preprocess_image(file)

#         # Save the final processed image
#         output_path = save_image(final_image, profile, file)
        
#         processed_images[file] = output_path  # Store the saved image path

#     return processed_images



