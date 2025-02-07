import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load bands B3 (Red), B2 (Green), and B1 (Blue)
band3 = rasterio.open(r"C:\Users\Asus\Desktop\satillite\2015\LC08_L2SP_225064_20150816_20200908_02_T1_SR_B3.TIF").read(1)
band2 = rasterio.open(r"C:\Users\Asus\Desktop\satillite\2015\LC08_L2SP_225064_20150816_20200908_02_T1_SR_B2.TIF").read(1)
band1 = rasterio.open(r"C:\Users\Asus\Desktop\satillite\2015\LC08_L2SP_225064_20150816_20200908_02_T1_SR_B1.TIF").read(1)

# Stack bands together (Red, Green, Blue)
rgb_image = np.dstack((band3, band2, band1))

# Normalize values for display (optional, improves visualization)
rgb_image = rgb_image / np.percentile(rgb_image, 98)  # Adjust brightness

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(rgb_image)
plt.axis('off')
plt.title("True Color Composite (B3, B2, B1)")
plt.show()
