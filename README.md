
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
all the other files in this repo are useful for those who are collecting data from Google Earth Engine (GEE),USGS (United States Geological Survey).Avoid them if you are collecting data from TerraBrasilis (INPE - Brazilian National Institute for Space Research) 
