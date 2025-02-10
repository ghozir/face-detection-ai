import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import json
from PIL import Image

# Path dataset & output model
DATASET_PATH = "./dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")
MODEL_DIR = "./app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "stress_detection_model.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# Pastikan folder models ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Image parameters
IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 32
EPOCHS = 10

# Data augmentation & preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Mapping kelas FER2013 ke stres level yang kita pakai
FER_TO_STRESS = {
    "angry": "stres_sangat_tinggi",
    "happy": "stres_rendah",
    "neutral": "stres_rendah",
    "sad": "stres_tinggi",
    "surprise": "stres_sedang"
}

# Filter kelas agar sesuai dengan mapping kita
filtered_classes = {k: v for k, v in train_generator.class_indices.items() if k in FER_TO_STRESS}
sorted_classes = sorted(filtered_classes.keys(), key=lambda x: ["angry", "happy", "neutral", "sad", "surprise"].index(x))

# Update class indices agar sesuai dengan urutan yang benar
class_indices = {label: idx for idx, label in enumerate(sorted_classes)}

# Simpan class indices untuk inference nanti
with open(LABELS_PATH, "w") as f:
    json.dump(class_indices, f)

print("Class mapping:", class_indices)

# Build CNN model
num_classes = len(class_indices)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

# Save model
model.save(MODEL_PATH)
print(f"Training completed and model saved as {MODEL_PATH}")

# Load trained model
model = load_model(MODEL_PATH)

# Load class indices
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
    stress_labels = {v: k for k, v in class_indices.items()}

# Function to predict stress level from an image
def predict_stress(image_path):
    image = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = stress_labels[predicted_class]
    
    print(f"Predicted stress level: {FER_TO_STRESS[predicted_label]}")
    return FER_TO_STRESS[predicted_label]

# Predict uploaded image
uploaded_image_path = "./image.jpg"
predict_stress(uploaded_image_path)
