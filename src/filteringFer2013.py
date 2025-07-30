import os
import shutil
import numpy as np
import tensorflow as tf
import cv2

# ========== CONFIG ==========
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.50
INPUT_BASE = 'datasets/fer2013/test'
OUTPUT_BASE = 'datasets/filterTestIterasi3'
MODEL_PATH = 'models/fineTunedModel_val0_6619_acc0_6465.h5'

# ========== LOAD MODEL ==========
model = tf.keras.models.load_model(MODEL_PATH)

# Kelas target yang ingin diproses
target_classes = ['angry', 'disgust', 'fear','surprise']

# Urutan klasifikasi sesuai pelatihan
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
class_name_to_idx = {name: i for i, name in enumerate(class_names)}

# ========== IMAGE PREPROCESS ==========
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 64, 64, 1)
    return img

# ========== PROSES ==========
total = 0
filtered = 0
misclassified = 0

for label_folder in os.listdir(INPUT_BASE):
    label_path = os.path.join(INPUT_BASE, label_folder)
    if not os.path.isdir(label_path):
        continue

    # Hanya proses folder yang termasuk dalam target
    if label_folder not in target_classes:
        continue

    for filename in os.listdir(label_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_path = os.path.join(label_path, filename)
        image = preprocess_image(file_path)

        if image is None:
            print(f"❌ Gagal baca: {filename}")
            continue

        preds = model.predict(image, verbose=0)
        pred_idx = np.argmax(preds)
        confidence = preds[0][pred_idx]
        pred_label = class_names[pred_idx]

        # Filter hanya jika prediksi sesuai label asli, confidence tinggi, dan label termasuk target
        if (
            pred_label == label_folder and
            confidence >= CONFIDENCE_THRESHOLD and
            pred_label in target_classes
        ):
            save_dir = os.path.join(OUTPUT_BASE, pred_label)
            os.makedirs(save_dir, exist_ok=True)
            shutil.copy(file_path, os.path.join(save_dir, filename))
            print(f"✅ {filename} VALID: {label_folder} == {pred_label} ({confidence*100:.2f}%)")
            filtered += 1
        else:
            print(f"❌ {filename} SKIPPED: real={label_folder}, pred={pred_label} ({confidence*100:.2f}%)")
            misclassified += 1

        total += 1

# ========== RINGKASAN ==========
print("\n📊 SUMMARY")
print(f"🎯 Total gambar diproses: {total}")
print(f"📂 Total valid & disalin: {filtered}")
print(f"⚠️ Salah prediksi atau confidence rendah: {misclassified}")
