import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = r"C:\Users\siva\Desktop\VIT\CNN\paddy_banana_cnn.keras"
TEST_DIR = r"C:\Users\siva\Desktop\VIT\CNN\TEST"

IMG_SIZE = (128, 128)

# Same mapping used during training
CLASS_NAMES = {
    0: "BANANA",
    1: "PADDY"
}

# -----------------------------
# LOAD MODEL
# -----------------------------
print("=" * 50)
print("LOADING CNN MODEL")
print("=" * 50)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# -----------------------------
# PREDICT EACH IMAGE
# -----------------------------
print("\n" + "=" * 50)
print("TEST IMAGE PREDICTIONS")
print("=" * 50)

image_files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if len(image_files) == 0:
    print("No images found in TEST folder.")
    exit()

for filename in sorted(image_files):

    image_path = os.path.join(TEST_DIR, filename)

    # Load image
    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize exactly like training
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    probability = float(model.predict(img_array, verbose=0)[0][0])

    # Keras mapping:
    # 0 = BANANA
    # 1 = PADDY

    if probability >= 0.5:
        prediction = "PADDY"
        confidence = probability
    else:
        prediction = "BANANA"
        confidence = 1 - probability

    print(
        f"{filename:30s} → "
        f"{prediction:7s} "
        f"Confidence: {confidence * 100:.2f}%"
    )