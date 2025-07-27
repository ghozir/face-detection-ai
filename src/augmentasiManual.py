import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# ====== CONFIGURABLE PARAMETERS ======
SOURCE_DIR = 'datasets/train'  # folder asli RAF-DB
TARGET_DIR = 'datasets_augmented/train'  # folder hasil augmentasi
AUG_PER_IMAGE = 5  # setiap gambar akan digandakan 5x
IMG_SIZE = (64, 64)  # ukuran output
SEED = 42

# ====== AUGMENTATION SETUP ======
augmentor = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5, 1.5),
    fill_mode='nearest'
)

# ====== CREATE TARGET FOLDERS ======
os.makedirs(TARGET_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    target_class_path = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🔄 Augmenting class '{class_name}' with {len(images)} images...")

    for img_name in tqdm(images):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)  # channel
        img = np.expand_dims(img, axis=0)   # batch

        prefix = os.path.splitext(img_name)[0]
        for i, aug in enumerate(augmentor.flow(img, batch_size=1, seed=SEED)):
            aug_img = aug[0].astype(np.uint8).squeeze()
            out_path = os.path.join(target_class_path, f"{prefix}_aug{i+1}.jpg")
            cv2.imwrite(out_path, aug_img)

            if i + 1 >= AUG_PER_IMAGE:
                break

print("✅ Semua gambar sudah digandakan dan disimpan.")
