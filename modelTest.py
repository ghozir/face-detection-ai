import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
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

# Load trained model
model = load_model(MODEL_PATH)

# Pastikan folder models ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameter model
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Mapping kelas FER2013 ke stres level
FER_TO_STRESS = {
    "angry": "stres_sangat_tinggi",
    "happy": "stres_rendah",
    "neutral": "stres_rendah",
    "sad": "stres_tinggi",
    "surprise": "stres_sedang"
}


# Load class indices
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
    stress_labels = {v: k for k, v in class_indices.items()}

# Fungsi prediksi tingkat stres
def predict_stress(image_path):
    image = load_img(image_path, target_size=IMG_SIZE, color_mode='grayscale')
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = stress_labels[predicted_class]
    
    print(f"Predicted stress level: {FER_TO_STRESS[predicted_label]}")
    return FER_TO_STRESS[predicted_label]

# Contoh prediksi
test_image_path = "./image.jpg"
predict_stress(test_image_path)