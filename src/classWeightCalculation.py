# ==================== CLASS WEIGHT & DISTRIBUSI KELAS ====================
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# Konfigurasi ulang datagen agar tidak perlu augmentasi
datagen_simple = ImageDataGenerator(rescale=1./255)

train_gen_for_weight = datagen_simple.flow_from_directory(
    'datasets/fer2013/train',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False  # penting agar urutan label tidak berubah
)

# Ambil label dari data training
y_labels = train_gen_for_weight.classes
class_indices = train_gen_for_weight.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

# Hitung distribusi kelas
counter = Counter(y_labels)
total_data = len(y_labels)

print("\n📊 Distribusi Jumlah Data per Kelas:")
print(f"  Total data: {total_data} gambar\n")
for class_idx in sorted(counter.keys()):
    class_name = inv_class_indices[class_idx]
    count = counter[class_idx]
    percentage = (count / total_data) * 100
    print(f"  - Class '{class_name}' (index {class_idx}): {count} gambar ({percentage:.2f}%)")

# Hitung class weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weights_dict = dict(enumerate(class_weights))

print("\n⚖️  Hasil Perhitungan Class Weights:")
for class_idx, weight in class_weights_dict.items():
    class_name = inv_class_indices[class_idx]
    print(f"  - Class '{class_name}' (index {class_idx}): {weight:.2f}")
