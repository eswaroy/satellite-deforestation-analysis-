
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the path to your file (change this to match your system)
# file_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"

# # Open the satellite image
# with rasterio.open(file_path) as dataset:
#     # Print available band count to verify
#     print("Number of Bands in Image:", dataset.count)
    
#     # Read Red and NIR bands (Landsat 8: Band 4 is Red, Band 5 is NIR)
#     red = dataset.read(4).astype(float)  # Band 4 (Red)
#     nir = dataset.read(5).astype(float)  # Band 5 (NIR)

#     # Avoid division by zero
#     ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

#     # Display NDVI Map
#     plt.figure(figsize=(8, 6))
#     plt.imshow(ndvi, cmap='RdYlGn')
#     plt.colorbar(label="NDVI")
#     plt.title("NDVI Map of Rondonia")
#     plt.show()
    
#     # Save NDVI as GeoTIFF
#     ndvi_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_NDVI.tif"
#     with rasterio.open(
#         ndvi_path, "w",
#         driver="GTiff",
#         height=ndvi.shape[0],
#         width=ndvi.shape[1],
#         count=1,  # Single-band output
#         dtype=rasterio.float32,
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(ndvi.astype(rasterio.float32), 1)

#     print(f"NDVI file saved at: {ndvi_path}")
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the path to your file (change this to match your system)
# file_path = r"C:\Users\Asus\Desktop\satillite\2024\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"

# # Open the satellite image
# with rasterio.open(file_path) as dataset:
#     # Print available band count to verify
#     print("Number of Bands in Image:", dataset.count)
    
#     # Read Red and NIR bands (Landsat 8: Band 4 is Red, Band 5 is NIR)
#     red = dataset.read(4).astype(float)  # Band 4 (Red)
#     nir = dataset.read(5).astype(float)  # Band 5 (NIR)

#     # Avoid division by zero
#     ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

#     # Threshold NDVI for deforestation detection
#     deforestation_mask = ndvi < 0.2  # Deforested areas (NDVI < 0.2)
    
#     # Display NDVI Map
#     plt.figure(figsize=(8, 6))
#     plt.imshow(ndvi, cmap='RdYlGn')
#     plt.colorbar(label="NDVI")
#     plt.title("NDVI Map of Rondonia")
#     plt.show()
    
#     # Display Deforestation Map
#     plt.figure(figsize=(8, 6))
#     plt.imshow(deforestation_mask, cmap='gray')
#     plt.title("Deforestation Mask (NDVI < 0.2)")
#     plt.show()
    
#     # Save NDVI as GeoTIFF
#     ndvi_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_NDVI.tif"
#     deforestation_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Deforestation.tif"
    
#     with rasterio.open(
#         ndvi_path, "w",
#         driver="GTiff",
#         height=ndvi.shape[0],
#         width=ndvi.shape[1],
#         count=1,  # Single-band output
#         dtype=rasterio.float32,
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(ndvi.astype(rasterio.float32), 1)
#     print(f"NDVI file saved at: {ndvi_path}")
    
#     # Save Deforestation Mask as GeoTIFF
#     with rasterio.open(
#         deforestation_path, "w",
#         driver="GTiff",
#         height=deforestation_mask.shape[0],
#         width=deforestation_mask.shape[1],
#         count=1,  # Single-band output
#         dtype=rasterio.uint8,  # Binary mask (0 or 1)
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(deforestation_mask.astype(rasterio.uint8), 1)
#     print(f"Deforestation mask saved at: {deforestation_path}")
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Define the path to your file (change this to match your system)
# file_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"

# # Open the satellite image
# with rasterio.open(file_path) as dataset:
#     # Print available band count to verify
#     print("Number of Bands in Image:", dataset.count)
    
#     # Read all spectral bands (Bands 1 to 7)
#     bands = []
#     for i in range(1, 8):  # Landsat 8 has Bands 1 to 7
#         bands.append(dataset.read(i).astype(float))
    
#     # Stack bands into a single array (shape: height x width x bands)
#     img_stack = np.stack(bands, axis=-1)
#     original_shape = img_stack.shape  # Save original shape for later
    
#     # Reshape image for clustering (pixels x features)
#     img_reshaped = img_stack.reshape(-1, 7)
    
#     # Apply K-Means Clustering (Land Cover Classification)
#     n_clusters = 5  # Number of land cover types (forest, water, urban, etc.)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clustered_img = kmeans.fit_predict(img_reshaped)
    
#     # Reshape back to original image shape
#     clustered_img = clustered_img.reshape(original_shape[:2])
    
#     # Display the clustered image
#     plt.figure(figsize=(8, 6))
#     plt.imshow(clustered_img, cmap='tab10')  # Use a categorical colormap
#     plt.colorbar(label="Land Cover Classes")
#     plt.title("K-Means Land Cover Classification")
#     plt.show()
    
#     # Save the clustered image as GeoTIFF
#     classified_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Clustered.tif"
#     with rasterio.open(
#         classified_path, "w",
#         driver="GTiff",
#         height=clustered_img.shape[0],
#         width=clustered_img.shape[1],
#         count=1,  # Single-band output
#         dtype=rasterio.uint8,
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(clustered_img.astype(rasterio.uint8), 1)
#     print(f"Clustered image saved at: {classified_path}")
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # Define the path to your file (change this to match your system)
# file_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"

# # Open the satellite image
# with rasterio.open(file_path) as dataset:
#     print("Number of Bands in Image:", dataset.count)
    
#     # Read all spectral bands (Bands 1 to 7)
#     bands = [dataset.read(i).astype(float) for i in range(1, 8)]
#     img_stack = np.stack(bands, axis=-1)
#     original_shape = img_stack.shape  
#     img_reshaped = img_stack.reshape(-1, 7)
#     img_reshaped = np.nan_to_num(img_reshaped, nan=np.nanmean(img_reshaped))  # Replace NaNs with mean value

    
#     # Apply K-Means Clustering for land cover classification
#     n_clusters = 5
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clustered_img = kmeans.fit_predict(img_reshaped)
#     clustered_img = clustered_img.reshape(original_shape[:2])
    
#     # Display the clustered image
#     plt.figure(figsize=(8, 6))
#     plt.imshow(clustered_img, cmap='tab10')
#     plt.colorbar(label="Land Cover Classes")
#     plt.title("K-Means Land Cover Classification")
#     plt.show()
    
#     # Save clustered image
#     classified_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Clustered.tif"
#     with rasterio.open(classified_path, "w", driver="GTiff", height=clustered_img.shape[0],
#                         width=clustered_img.shape[1], count=1, dtype=rasterio.uint8,
#                         crs=dataset.crs, transform=dataset.transform) as dst:
#         dst.write(clustered_img.astype(rasterio.uint8), 1)
#     print(f"Clustered image saved at: {classified_path}")

# # U-Net Model for Forest Loss Detection
# def unet_model(input_shape=(256, 256, 3)):
#     inputs = Input(input_shape)
    
#     # Encoder
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)
    
#     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
#     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)
    
#     c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
#     c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
#     p3 = MaxPooling2D((2, 2))(c3)
    
#     # Bottleneck
#     c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
#     c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
#     # Decoder
#     u1 = UpSampling2D((2, 2))(c4)
#     u1 = concatenate([u1, c3])
#     c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
#     c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
#     u2 = UpSampling2D((2, 2))(c5)
#     u2 = concatenate([u2, c2])
#     c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
#     c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
#     u3 = UpSampling2D((2, 2))(c6)
#     u3 = concatenate([u3, c1])
#     c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
#     c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
#     model = Model(inputs, outputs)
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model

# # Define U-Net model
# model = unet_model()
# model.summary()

# # Next step: Load training data and train the model
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Define file paths (Change these paths accordingly)
# file_path = r"C:\Users\Asus\Desktop\arf satillite-details\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"
# ndvi_output_path = r"C:\Users\Asus\Desktop\arf satillite-details\2015\ndvi_output.tif"
# mask_output_path = r"C:\Users\Asus\Desktop\arf satillite-details\2015\Rondonia_Deforestation_Mask_output.tif"
# classified_output_path = r"C:\Users\Asus\Desktop\arf satillite-details\2015\Rondonia_Clustered_output.tif"

# # Open the satellite image
# with rasterio.open(file_path) as dataset:
#     print("Number of Bands in Image:", dataset.count)
    
#     # Read spectral bands (Landsat 8 has bands 1 to 7)
#     bands = [dataset.read(i).astype(float) for i in range(1, 8)]
    
#     # Stack into a single array (height x width x bands)
#     img_stack = np.stack(bands, axis=-1)
    
#     # Compute NDVI: (NIR - Red) / (NIR + Red)
#     nir = dataset.read(5).astype(float)  # Band 5 (Near-Infrared)
#     red = dataset.read(4).astype(float)  # Band 4 (Red)
    
#     ndvi = (nir - red) / (nir + red + 1e-10)  # Adding small value to avoid division by zero
    
#     # Normalize NDVI for visualization (optional)
#     ndvi_normalized = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi))

#     # Apply Thresholding to Create a Binary Mask (Deforestation Detection)
#     ndvi_threshold = 0.2
#     print("NDVI Min:", np.min(ndvi))
#     print("NDVI Max:", np.max(ndvi))
#     print("Unique NDVI values:", np.unique(ndvi))
#     print("NDVI Min:", np.nanmin(ndvi), "NDVI Max:", np.nanmax(ndvi))
#     print("Total NaN Pixels:", np.sum(np.isnan(ndvi)))

#     ndvi_threshold = 0.2  # Best threshold for deforestation detection
#     valid_ndvi = np.where(np.isnan(ndvi), np.nan, ndvi)  # Keep NaNs
#     # valid_ndvi = np.where(ndvi != -1, ndvi, np.nan)  # Ignore replaced -1 values
#     deforestation_mask = (valid_ndvi < ndvi_threshold).astype(np.uint8)  # 1 = deforested, 0 = forested

#     # Display NDVI
#     plt.figure(figsize=(8, 6))
#     plt.imshow(ndvi, cmap='RdYlGn')  # Green = High NDVI, Red = Low NDVI
#     plt.colorbar(label="NDVI")
#     plt.title("NDVI Image")
#     plt.show()

#     # Display Deforestation Mask
#     plt.figure(figsize=(8, 6))
#     plt.imshow(deforestation_mask, cmap='gray')  # Black = Forest, White = Deforested
#     plt.colorbar(label="Deforestation Mask")
#     plt.title("Deforestation Detection Mask")
#     plt.show()

#     # Save NDVI as GeoTIFF
#     with rasterio.open(
#         ndvi_output_path, "w",
#         driver="GTiff",
#         height=ndvi.shape[0],
#         width=ndvi.shape[1],
#         count=1,
#         dtype=rasterio.float32,
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(ndvi.astype(rasterio.float32), 1)
#     print(f"NDVI image saved at: {ndvi_output_path}")

#     # Save Deforestation Mask as GeoTIFF
#     with rasterio.open(
#         mask_output_path, "w",
#         driver="GTiff",
#         height=deforestation_mask.shape[0],
#         width=deforestation_mask.shape[1],
#         count=1,
#         dtype=rasterio.uint8,
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(deforestation_mask, 1)
#     print(f"Deforestation mask saved at: {mask_output_path}")

#     # Apply K-Means Clustering (Land Cover Classification)
#     img_reshaped = img_stack.reshape(-1, 7)  # Reshape for clustering
#     n_clusters = 5
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clustered_img = kmeans.fit_predict(img_reshaped)
#     clustered_img = clustered_img.reshape(ndvi.shape)

#     # Save the clustered image as GeoTIFF
#     with rasterio.open(
#         classified_output_path, "w",
#         driver="GTiff",
#         height=clustered_img.shape[0],
#         width=clustered_img.shape[1],
#         count=1,
#         dtype=rasterio.uint8,
#         crs=dataset.crs,
#         transform=dataset.transform
#     ) as dst:
#         dst.write(clustered_img.astype(rasterio.uint8), 1)
#     print(f"Clustered image saved at: {classified_output_path}")
# print("NDVI Min:", np.min(ndvi))
# print("NDVI Max:", np.max(ndvi))
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define file paths (Change these paths accordingly)
file_path = r"C:\Users\Asus\Desktop\arf satillite-details\2024\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"#"C:\Users\Asus\Desktop\arf satillite-details\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"
ndvi_output_path = r"C:\Users\Asus\Desktop\arf satillite-details\2024\ndvi_output.tif"
mask_output_path = r"C:\Users\Asus\Desktop\arf satillite-details\2024\Rondonia_Deforestation_Mask_output.tif"
classified_output_path = r"C:\Users\Asus\Desktop\arf satillite-details\2024\Rondonia_Clustered_output.tif"

# Open the satellite image
with rasterio.open(file_path) as dataset:
    print("Number of Bands in Image:", dataset.count)
    
    # Read spectral bands (Landsat 8 has bands 1 to 7)
    bands = [dataset.read(i).astype(float) for i in range(1, 8)]
    img_stack = np.stack(bands, axis=-1)

    # Compute NDVI: (NIR - Red) / (NIR + Red)
    nir = dataset.read(5).astype(float)  # Band 5 (Near-Infrared)
    red = dataset.read(4).astype(float)  # Band 4 (Red)
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(nir), where=(nir + red) != 0)

    
    # ndvi = (nir - red) / (nir + red + 1e-10)  # Avoid division by zero
    ndvi = np.nan_to_num(ndvi, nan=-1)  # Replace NaNs with -1

    # Print NDVI stats
    print("NDVI Min:", np.min(ndvi), "NDVI Max:", np.max(ndvi))
    print("Unique NDVI values:", np.unique(ndvi))
    print("Total NaN Pixels (before handling):", np.sum(np.isnan(ndvi)))

    # Apply Thresholding to Create a Binary Mask (Deforestation Detection)
    ndvi_threshold = 0.2
    deforestation_mask = np.where(ndvi >= -1, (ndvi < ndvi_threshold).astype(np.uint8), 0)

    # Normalize Mask for Saving (0 = forest, 255 = deforested)
    deforestation_mask = deforestation_mask * 255

    # Display NDVI
    plt.figure(figsize=(8, 6))
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar(label="NDVI")
    plt.title("NDVI Image")
    plt.show()

    # Display Deforestation Mask
    plt.figure(figsize=(8, 6))
    plt.imshow(deforestation_mask, cmap='gray')
    plt.colorbar(label="Deforestation Mask")
    plt.title("Deforestation Detection Mask")
    plt.show()

    # Save NDVI as GeoTIFF
    with rasterio.open(
        ndvi_output_path, "w",
        driver="GTiff",
        height=ndvi.shape[0],
        width=ndvi.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=dataset.crs,
        transform=dataset.transform
    ) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)
    print(f"NDVI image saved at: {ndvi_output_path}")

    # Save Deforestation Mask as GeoTIFF
    with rasterio.open(
        mask_output_path, "w",
        driver="GTiff",
        height=deforestation_mask.shape[0],
        width=deforestation_mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=dataset.crs,
        transform=dataset.transform
    ) as dst:
        dst.write(deforestation_mask, 1)
    print(f"Deforestation mask saved at: {mask_output_path}")

    # Apply K-Means Clustering (Land Cover Classification)
    img_reshaped = img_stack.reshape(-1, 7)

    # Remove rows containing NaN before clustering
    non_nan_pixels = img_reshaped[~np.isnan(img_reshaped).any(axis=1)]
    
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustered_values = kmeans.fit_predict(non_nan_pixels)

    # Reinsert NaNs as -1 in the reshaped array
    clustered_img = np.full(img_reshaped.shape[0], -1)
    clustered_img[~np.isnan(img_reshaped).any(axis=1)] = clustered_values
    clustered_img = clustered_img.reshape(ndvi.shape)

    # Save the clustered image as GeoTIFF
    with rasterio.open(
        classified_output_path, "w",
        driver="GTiff",
        height=clustered_img.shape[0],
        width=clustered_img.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=dataset.crs,
        transform=dataset.transform
    ) as dst:
        dst.write(clustered_img.astype(rasterio.uint8), 1)
    print(f"Clustered image saved at: {classified_output_path}")

print("✅ Processing Complete")



