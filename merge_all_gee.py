import rasterio
from rasterio.merge import merge
import glob

# Step 1: Find all .tif files in the folder
folder_path = r"C:\Users\Asus\Desktop\satillite\2015"
tif_files = glob.glob(folder_path +r"\*.tif")

# Step 2: Open and merge files
datasets = [rasterio.open(file) for file in tif_files]
merged_image, merged_transform = merge(datasets)

# Step 3: Save the merged image
output_path = "processed_output/cloud_masked"
with rasterio.open(output_path, "w", 
                   driver="GTiff", 
                   height=merged_image.shape[1],
                   width=merged_image.shape[2],
                   count=datasets[0].count, 
                   dtype=datasets[0].dtypes[0],
                   transform=merged_transform) as merged_dataset:
    merged_dataset.write(merged_image)

print("Merged file saved at:", output_path)
