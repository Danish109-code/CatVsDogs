import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from tensorflow.keras.models import load_model

# ==========================================
# Step 1 - Load Trained Model
# ==========================================

model = load_model(
    "models/cat_dog_model.keras"
)

print("Model Loaded Successfully!")

# ==========================================
# Step 2 - Image Path
# ==========================================

image_path = "test_images/3.jpg"

print("Testing Image:", image_path)

# ==========================================
# Step 3 - Load Image
# ==========================================

image = Image.open(
    image_path
)

print(
    "Original Image Size:",
    image.size
)

# ==========================================
# Step 4 - Display Image
# ==========================================

plt.imshow(image)

plt.title("Test Image")

plt.axis("off")

plt.show()

# ==========================================
# Step 5 - Resize Image
# ==========================================

image = image.resize(
    (224, 224)
)

print(
    "Resized Image Size:",
    image.size
)

# ==========================================
# Step 6 - Convert Image to NumPy Array
# ==========================================

image_array = np.array(
    image
)

print(
    "Image Array Shape:",
    image_array.shape
)

# ==========================================
# Step 7 - Normalize Image
# ==========================================

image_array = image_array / 255.0

# ==========================================
# Step 8 - Add Batch Dimension
# ==========================================

image_array = np.expand_dims(
    image_array,
    axis=0
)

print(
    "Final Image Shape:",
    image_array.shape
)


# ==========================================
# Step 9 - Make Prediction
# ==========================================

prediction = model.predict(
    image_array
)

print(
    "Raw Prediction:",
    prediction
)

# ==========================================
# Step 10 - Classify Image
# ==========================================

probability = prediction[0][0]

if probability < 0.5:

    label = "Cat 🐱"

    confidence = (1 - probability) * 100

else:

    label = "Dog 🐶"

    confidence = probability * 100

# ==========================================
# Step 11 - Display Prediction
# ==========================================

print(
    "Prediction:",
    label
)

print(
    f"Confidence: {confidence:.2f}%"
)    