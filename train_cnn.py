import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = r"C:\Users\siva\Desktop\VIT\CNN"

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 30

SEED = 42


# ============================================================
# 2. CHECK DATASET
# ============================================================

banana_dir = os.path.join(DATASET_DIR, "BANANA")
paddy_dir = os.path.join(DATASET_DIR, "PADDY")

print("========================================")
print("DATASET CHECK")
print("========================================")

print("Banana folder:", banana_dir)
print("Paddy folder :", paddy_dir)

banana_images = [
    f for f in os.listdir(banana_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

paddy_images = [
    f for f in os.listdir(paddy_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print("Banana images:", len(banana_images))
print("Paddy images :", len(paddy_images))
print("Total images :", len(banana_images) + len(paddy_images))


# ============================================================
# 3. DATA GENERATORS
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    validation_split=0.20,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True

)


test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# ============================================================
# 4. TRAINING DATA
# ============================================================

train_data = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True,

    seed=SEED

)


# ============================================================
# 5. VALIDATION DATA
# ============================================================

validation_data = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False,

    seed=SEED

)


# ============================================================
# 6. PRINT CLASS MAPPING
# ============================================================

print("\n========================================")
print("CLASS MAPPING")
print("========================================")

print(train_data.class_indices)


# ============================================================
# 7. BUILD CNN
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(128, 128, 3)
    ),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Reduce spatial dimensions
    layers.GlobalAveragePooling2D(),

    # Regularization
    layers.Dropout(0.40),

    # Fully connected layer
    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.30),

    # Binary output
    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

print("\n========================================")
print("CNN MODEL")
print("========================================")

model.summary()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========================================")
print("TRAINING CNN")
print("========================================")

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS

)


# ============================================================
# 11. VALIDATION EVALUATION
# ============================================================

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

loss, accuracy = model.evaluate(
    validation_data
)

print("Validation Loss    :", loss)
print("Validation Accuracy:", accuracy)


# ============================================================
# 12. PREDICTIONS
# ============================================================

validation_data.reset()

probabilities = model.predict(
    validation_data
)

predictions = (
    probabilities.ravel() >= 0.5
).astype(int)

actual = validation_data.classes


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual,
    predictions
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

class_names = list(
    validation_data.class_indices.keys()
)

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        actual,
        predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 15. PLOT TRAINING HISTORY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cnn_accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cnn_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. SAVE MODEL
# ============================================================

model.save(
    "paddy_banana_cnn.keras"
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    "Saved as: paddy_banana_cnn.keras"
)

print("\nCNN training completed.")