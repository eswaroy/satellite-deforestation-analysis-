import os
import numpy as np
import tensorflow as tf
import psutil  # 🔥 Monitor system temperature
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight

# ✅ Step 1: Load and Preprocess Data
DATASET_DIR = r"C:\Users\Asus\Desktop\arf satillite-details\aa"  # Path to PNG images
MODEL_PATH = "deforestation_model.h5"  # Model save path

BATCH_SIZE = 32
IMG_SIZE = (256, 256)

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # 20% for validation
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="binary",
    subset="validation"
)

# ✅ Step 2: Compute Class Weights (Handling Imbalance)
classes = np.array([0, 1])  # 0 = Non-Deforested, 1 = Deforested
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_data.classes)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class Weights:", class_weight_dict)

# ✅ Step 3: Define CNN Model with Transfer Learning
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)  # Resume training if model exists
else:
    print("Creating new model...")
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False  # Freeze the base model for feature extraction

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)  # Binary classification

    model = Model(inputs=base_model.input, outputs=output)

# ✅ Step 4: Compile Model with Focal Loss
focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
model.compile(optimizer=AdamW(learning_rate=0.0001), loss=focal_loss, metrics=["accuracy"])

# ✅ Step 5: Define Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

# ✅ Step 6: Custom Callback to Pause Training on Overheating
class HeatMonitorCallback(Callback):
    def __init__(self, max_temp=70):
        super().__init__()
        self.max_temp = max_temp

    def on_epoch_begin(self, epoch, logs=None):
        while self.get_cpu_temp() > self.max_temp:
            print(f"⚠️ CPU Overheated! ({self.get_cpu_temp()}°C) Pausing training...")
            time.sleep(60)  # Wait for 1 minute before checking again
        print(f"✅ CPU Temperature Normal ({self.get_cpu_temp()}°C), Resuming training...")

    def on_epoch_end(self, epoch, logs=None):
        model.save(MODEL_PATH)  # Save model after every epoch
        print(f"💾 Model saved after epoch {epoch+1}.")

    def get_cpu_temp(self):
        # Get system temperature (Windows/Linux)
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:  # Linux
                return temps["coretemp"][0].current
            elif "cpu_thermal" in temps:  # Some Raspberry Pi boards
                return temps["cpu_thermal"][0].current
            else:
                return 50  # Default safe temp if unknown
        return 50  # Default safe temp for Windows (no direct temperature access)

# ✅ Step 7: Train Model
EPOCHS = 30

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, HeatMonitorCallback()]
)

# ✅ Step 8: Save Model & Evaluate
model.save(MODEL_PATH)
print("✅ Final Model saved!")

test_loss, test_acc = model.evaluate(val_data)
print(f"🔍 Test Accuracy: {test_acc * 100:.2f}%")
