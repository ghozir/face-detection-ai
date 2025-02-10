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

# Pastikan folder models ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameter model
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Augmentasi data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
generator_train = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

generator_test = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Mapping kelas FER2013 ke stres level
FER_TO_STRESS = {
    "angry": "stres_sangat_tinggi",
    "happy": "stres_rendah",
    "neutral": "stres_rendah",
    "sad": "stres_tinggi",
    "surprise": "stres_sedang"
}

# Filter kelas agar sesuai dengan mapping kita
filtered_classes = {k: v for k, v in generator_train.class_indices.items() if k in FER_TO_STRESS}
sorted_classes = sorted(filtered_classes.keys(), key=lambda x: ["angry", "happy", "neutral", "sad", "surprise"].index(x))

# Update class indices
class_indices = {label: idx for idx, label in enumerate(sorted_classes)}

# Simpan class indices untuk inference
with open(LABELS_PATH, "w") as f:
    json.dump(class_indices, f)

print("Class mapping:", class_indices)

# Model CNN dengan BatchNormalization
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_indices), activation='softmax')
])

# Compile model dengan learning rate rendah
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Custom Data Generator untuk augmentasi tambahan
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator):
        self.generator = generator
    
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        images, labels = self.generator[index]
        augmented_images = np.array([train_datagen.random_transform(image) for image in images])
        images = np.concatenate([images, augmented_images], axis=0)
        labels = np.concatenate([labels, labels], axis=0)
        return images, labels

generator_train = CustomDataGenerator(generator_train)
generator_test = CustomDataGenerator(generator_test)

# Training model
print(f"Epochs yang digunakan: {EPOCHS}")
model.fit(generator_train, validation_data=generator_test, epochs=EPOCHS)

# Simpan model dalam format .keras
model.save(MODEL_PATH)
print(f"Training completed and model saved as {MODEL_PATH}")

# Load trained model
model = load_model(MODEL_PATH)

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
