
# Deforestation Analysis and Prediction

## Overview
This project aims to analyze and predict deforestation trends using satellite imagery and machine learning. The workflow includes image processing, dataset augmentation, and training a deep learning model to classify deforested areas.

## Project Structure
```
├── analyze_deforestation.py      # Analyzes deforestation from processed images
├── convert_tif_to_png.py         # Converts satellite TIFF images to PNG format
├── divide_image_to_tiles.py      # Splits large images into smaller tiles
├── graph.png                     # Visualization of deforestation change over time
├── ndvi_gee.py                   # Computes NDVI using TerraBrasilis (INPE - Brazilian National Institute for Space Research) 
├── normalize_augument_setdata.py  # Normalizes and augments dataset images
├── train_model.py                 # Trains a deep learning model for deforestation detection
```

## Features
- **Image Processing**: Converts and prepares satellite images for analysis.
- **NDVI Calculation**: Uses TerraBrasilis (INPE - Brazilian National Institute for Space Research)  to compute vegetation indices.
- **Data Augmentation**: Applies transformations to enhance dataset variability.
- **Deep Learning Model**: Uses EfficientNetB3 for deforestation classification.
- **Performance Monitoring**: Includes overheating protection for training stability.

## Requirements
Install the necessary dependencies before running the scripts:
```bash
pip install tensorflow numpy psutil scikit-learn matplotlib earthengine-api
```

## Usage
1. **Preprocess Images**
   - Convert TIFF images to PNG:
     ```bash
     python convert_tif_to_png.py
     ```
   - Split large images into tiles:
     ```bash
     python divide_image_to_tiles.py
     ```
   
2. **Compute NDVI using Google Earth Engine**
   ```bash
   python ndvi_gee.py
   ```

3. **Normalize and Augment Data**
   ```bash
   python normalize_augument_setdata.py
   ```

4. **Train the Model**
   ```bash
   python train_model.py
   ```

5. **Analyze Results**
   ```bash
   python analyze_deforestation.py
   ```

## Results
The project generates a **graph (graph.png)** showing deforestation trends over time.

## Contributors
- **Your Name** (Replace with your details)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
# Note
All the other files in this repo are useful for those who are collecting data from Google Earth Engine (GEE),USGS (United States Geological Survey).Avoid them if you are collecting data from TerraBrasilis (INPE - Brazilian National Institute for Space Research) 
# Detailed Description
# 🌿 Deforestation Analysis and Prediction using Satellite Imagery  

## 📌 Project Overview  
This project aims to **analyze and predict deforestation trends** in the Amazon rainforest using **satellite imagery, remote sensing techniques, and deep learning models**. By leveraging **Landsat data, NDVI calculations, and machine learning**, we classify deforested vs. non-deforested areas and track vegetation loss over time.  

## 🌍 Data Sources  
We collected and processed data from:  
- **USGS (United States Geological Survey)** – Landsat satellite images (2015 & 2024)  
- **TerraBrasilis (INPE - Brazilian Space Agency)** – PRODES deforestation monitoring dataset  
- **Google Earth Engine (GEE)** – NDVI calculation for vegetation analysis  
- **Custom Image Processing** – Image tiling, conversion, and augmentation for deep learning  

## 🔍 Key Features  
- **🛰️ Satellite Image Processing:** Converts TIFF images to PNG and extracts meaningful data  
- **🌱 NDVI Calculation:** Uses remote sensing techniques to detect vegetation health  
- **🧩 Image Tiling & Augmentation:** Improves model training efficiency and accuracy  
- **🧠 Deep Learning Model:** Utilizes **EfficientNetB3** for deforestation classification  
- **📊 Data Visualization:** Generates deforestation trend graphs for better insights  
- **🚀 Performance Optimization:** Includes overheating protection for stable training  

## 🛠️ Technologies Used  
- **Python** (TensorFlow, OpenCV, NumPy, Matplotlib, Rasterio, Scikit-learn)  
- **Google Earth Engine API** for satellite image analysis  
- **Deep Learning** (EfficientNetB3, CNN-based classification)  
- **Geospatial Processing** (GDAL, Rasterio)  

## 📊 Expected Results  
- Identification of deforested areas from satellite imagery  
- Prediction of forest loss trends over time  
- Visualization of deforestation rates in different regions  

## 🤝 Contributors  
---No Contributors----
-Dasari Ranga Eswar[(Me)one man show]

## 📜 License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  
