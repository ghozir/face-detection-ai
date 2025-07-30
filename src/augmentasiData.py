import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import os

# ======== Konfigurasi ========
IMG_PATH = 'datasets/raf-db-original/train/sad/train_00025_aligned.jpg'
IMG_SIZE = (64, 64)
SAVE_PATH = "logs/hasil_augmentasi_grayscale.png"
os.makedirs("logs", exist_ok=True)

# ======== Load Gambar RGB (tanpa normalisasi) ========
img = load_img(IMG_PATH, color_mode='rgb', target_size=IMG_SIZE)
img_array = img_to_array(img)  # dtype=uint8

# Simpan original grayscale juga buat perbandingan
original_gray = np.mean(img_array, axis=-1) / 255.0

# Expand dimensi buat flow
img_array_exp = np.expand_dims(img_array, axis=0)

# ======== Augmentasi Generator ========
augment_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

aug_iter = augment_datagen.flow(img_array_exp, batch_size=1)
augmented_rgb = next(aug_iter)[0]  # Masih uint8

# ======== Konversi ke grayscale setelah augmentasi ========
augmented_gray = np.mean(augmented_rgb, axis=-1) / 255.0  # (64, 64), float32 [0–1]

# ======== Tampilkan dan Simpan ========
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].imshow(original_gray, cmap='gray')
axs[0].set_title("Original Grayscale")
axs[0].axis("off")

axs[1].imshow(augmented_gray, cmap='gray')
axs[1].set_title("Augmented Grayscale")
axs[1].axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH)

print(f"✅ Saved as {SAVE_PATH}")
print("Augmented shape:", augmented_gray.shape)
print("Augmented min/max:", augmented_gray.min(), augmented_gray.max())
print("Augmented mean:", np.mean(augmented_gray))