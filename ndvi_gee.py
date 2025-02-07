
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
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the path to your file (change this to match your system)
file_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Landsat8_Bands1to7_CloudMasked-0000000000-0000000000.tif"

# Open the satellite image
with rasterio.open(file_path) as dataset:
    print("Number of Bands in Image:", dataset.count)
    
    # Read all spectral bands (Bands 1 to 7)
    bands = [dataset.read(i).astype(float) for i in range(1, 8)]
    img_stack = np.stack(bands, axis=-1)
    original_shape = img_stack.shape  
    img_reshaped = img_stack.reshape(-1, 7)
    img_reshaped = np.nan_to_num(img_reshaped, nan=np.nanmean(img_reshaped))  # Replace NaNs with mean value

    
    # Apply K-Means Clustering for land cover classification
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustered_img = kmeans.fit_predict(img_reshaped)
    clustered_img = clustered_img.reshape(original_shape[:2])
    
    # Display the clustered image
    plt.figure(figsize=(8, 6))
    plt.imshow(clustered_img, cmap='tab10')
    plt.colorbar(label="Land Cover Classes")
    plt.title("K-Means Land Cover Classification")
    plt.show()
    
    # Save clustered image
    classified_path = r"C:\Users\Asus\Desktop\satillite\2015\Rondonia_Clustered.tif"
    with rasterio.open(classified_path, "w", driver="GTiff", height=clustered_img.shape[0],
                        width=clustered_img.shape[1], count=1, dtype=rasterio.uint8,
                        crs=dataset.crs, transform=dataset.transform) as dst:
        dst.write(clustered_img.astype(rasterio.uint8), 1)
    print(f"Clustered image saved at: {classified_path}")

# U-Net Model for Forest Loss Detection
def unet_model(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder
    u1 = UpSampling2D((2, 2))(c4)
    u1 = concatenate([u1, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
    u2 = UpSampling2D((2, 2))(c5)
    u2 = concatenate([u2, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
    u3 = UpSampling2D((2, 2))(c6)
    u3 = concatenate([u3, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Define U-Net model
model = unet_model()
model.summary()

# Next step: Load training data and train the model



