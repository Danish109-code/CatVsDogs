import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from tensorflow.keras.models import load_model

import os
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ==========================================
# Step 1 - Dataset Path
# ==========================================

dataset_path = "dataset"

print("Dataset folders:")
print(os.listdir(dataset_path))


# ==========================================
# Step 2 - Count Images
# ==========================================

cats_folder = os.path.join(
    dataset_path,
    "Cat"
)

dogs_folder = os.path.join(
    dataset_path,
    "Dog"
)

cat_count = len(
    os.listdir(cats_folder)
)

dog_count = len(
    os.listdir(dogs_folder)
)

print("Cats :", cat_count)
print("Dogs :", dog_count)


# ==========================================
# Step 3 - Dataset Distribution
# ==========================================

labels = ["Cats", "Dogs"]

counts = [
    cat_count,
    dog_count
]

plt.figure(figsize=(6, 5))

plt.bar(
    labels,
    counts,
    color=["orange", "blue"]
)

plt.title("Dataset Distribution")

plt.xlabel("Classes")

plt.ylabel("Number of Images")

plt.show()


# ==========================================
# Step 4 - Display Sample Images
# ==========================================

cat_image_name = random.choice(
    os.listdir(cats_folder)
)

dog_image_name = random.choice(
    os.listdir(dogs_folder)
)

cat_image_path = os.path.join(
    cats_folder,
    cat_image_name
)

dog_image_path = os.path.join(
    dogs_folder,
    dog_image_name
)

cat_image = Image.open(
    cat_image_path
)

dog_image = Image.open(
    dog_image_path
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.imshow(cat_image)

plt.title("Cat")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(dog_image)

plt.title("Dog")

plt.axis("off")

plt.show()


# ==========================================
# Step 5 - Data Augmentation
# ==========================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    rotation_range=20,

    zoom_range=0.20,

    width_shift_range=0.20,

    height_shift_range=0.20,

    horizontal_flip=True,

    validation_split=0.20
)


# ==========================================
# Step 6 - Training Generator
# ==========================================

train_generator = train_datagen.flow_from_directory(

    directory=dataset_path,

    target_size=(224, 224),

    batch_size=32,

    class_mode="binary",

    subset="training",

    shuffle=True
)


# ==========================================
# Step 7 - Validation Generator
# ==========================================

validation_generator = train_datagen.flow_from_directory(

    directory=dataset_path,

    target_size=(224, 224),

    batch_size=32,

    class_mode="binary",

    subset="validation",

    shuffle=False
)


# ==========================================
# Step 8 - Class Mapping
# ==========================================

print(
    "Class Indices:",
    train_generator.class_indices
)

print(
    "Training Images:",
    train_generator.samples
)

print(
    "Validation Images:",
    validation_generator.samples
)


# ==========================================
# Step 9 - Check Image Shape
# ==========================================

images, labels = next(
    train_generator
)

print(
    "Images Shape:",
    images.shape
)

print(
    "Labels Shape:",
    labels.shape
)


# ==========================================
# Step 10 - Display Training Images
# ==========================================

images, labels = next(
    train_generator
)

plt.figure(figsize=(12, 8))

for i in range(9):

    plt.subplot(
        3,
        3,
        i + 1
    )

    plt.imshow(
        images[i]
    )

    if int(labels[i]) == 0:

        class_name = "Cat"

    else:

        class_name = "Dog"

    plt.title(
        f"Label: {class_name}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()


# ==========================================
# Step 11 - Build CNN Model
# ==========================================

model = Sequential([

    Input(
        shape=(224, 224, 3)
    ),

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.5
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# Step 12 - Model Summary
# ==========================================

model.summary()


# ==========================================
# Step 13 - Compile Model
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

print(
    "Model compiled successfully."
)


# ==========================================
# Step 14 - Create Models Folder
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# Step 15 - Early Stopping
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


# ==========================================
# Step 16 - Model Checkpoint
# ==========================================

checkpoint = ModelCheckpoint(

    "models/cat_dog_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ==========================================
# Step 17 - Train Model
# ==========================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=5,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)

print(
    "\nTraining Completed Successfully!"
)


# ==========================================
# Step 18 - Accuracy Graph
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    color="blue"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    color="red"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "models/accuracy_graph.png"
)

plt.show()


# ==========================================
# Step 19 - Loss Graph
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    color="blue"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    color="red"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "models/loss_graph.png"
)

plt.show()


# ==========================================
# Step 20 - Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(
    validation_generator
)

print(
    "\nValidation Loss:",
    loss
)

print(
    "Validation Accuracy:",
    accuracy
)


# ==========================================
# Step 21 - Load Best Model
# ==========================================

loaded_model = load_model(
    "models/cat_dog_model.keras"
)

print(
    "\nBest model loaded successfully!"
)

loaded_model.summary()