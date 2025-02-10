import tensorflow as tf
import numpy as np
import os
import json
import cv2
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

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

# Augmentasi manual
ROTATION_RANGE = 20
NOISE_STD_DEV = 0.05
BRIGHTNESS_VARIATION = 30

def augment_image(image):
    """Melakukan augmentasi pada gambar."""
    # Konversi ke OpenCV format
    image = (image * 255).astype(np.uint8)
    
    # Rotasi Acak
    angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
    center = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (IMG_WIDTH, IMG_HEIGHT), borderMode=cv2.BORDER_REFLECT)
    
    # Gaussian Noise
    noise = np.random.normal(0, NOISE_STD_DEV * 255, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Brightness Adjustment
    # Brightness augmentation
    brightness_factor = np.random.randint(-30, 30)
    image = image.astype(np.int16)  # Ubah ke int16 dulu untuk mencegah overflow
    image = np.clip(image + brightness_factor, 0, 255)  # Pastikan tetap dalam range uint8
    image = image.astype(np.uint8)  # Kembalikan ke uint8
    
    # # Normalize back to [0,1]
    # image = image.astype(np.float32) / 255.0
    return image

class AugmentedImageDataGenerator(Sequence):
    def __init__(self, directory, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        self.filepaths, self.labels = self._load_data(directory)
        self.num_classes = len(set(self.labels))
        
    def _load_data(self, directory):
        """Membaca path gambar dan labelnya."""
        filepaths = []
        labels = []
        class_indices = {cls: idx for idx, cls in enumerate(sorted(os.listdir(directory)))}
        for cls in class_indices:
            cls_dir = os.path.join(directory, cls)
            for filename in os.listdir(cls_dir):
                filepaths.append(os.path.join(cls_dir, filename))
                labels.append(class_indices[cls])
        return filepaths, labels
    
    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_paths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        images = np.zeros((len(batch_paths), self.img_size[0], self.img_size[1], 1))
        labels = np.zeros((len(batch_paths), self.num_classes))
        
        for i, path in enumerate(batch_paths):
            image = load_img(path, target_size=self.img_size, color_mode='grayscale')
            image = img_to_array(image) / 255.0
            image = augment_image(image)
            images[i] = np.expand_dims(image, axis=-1)
            labels[i][batch_labels[i]] = 1  # One-hot encoding
        
        return images, labels

# Buat generator untuk training dan testing
generator_train = AugmentedImageDataGenerator(TRAIN_DIR, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))
generator_test = AugmentedImageDataGenerator(TEST_DIR, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))

# Build CNN model
num_classes = len(set(generator_train.labels))
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
model.fit(generator_train, validation_data=generator_test, epochs=EPOCHS)

# Save model
model.save(MODEL_PATH)
print(f"Training completed and model saved as {MODEL_PATH}")
