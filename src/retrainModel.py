import os
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam

# ==================== CONFIG ====================
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 100 
MODEL_PATH = 'models/bestModel_val0_6722_acc0_6406.h5'

# Dataset baru hasil gabungan atau subset dari FER2013
TRAIN_PATH = 'datasets/raf-db-iterasi2/train'
VAL_PATH = 'datasets/raf-db-iterasi2/test'  # atau test dari rafdb

# ==================== LOAD PRETRAINED MODEL ====================
print("📥 Loading pretrained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Optional: freeze layer awal kalau kamu mau
for layer in model.layers[:10]:  # contoh freeze 10 layer pertama
    layer.trainable = False

# ==================== RE-COMPILE ====================
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # fine-tune → learning rate kecil
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ==================== DATASET LOADER ====================
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

datagen_test = ImageDataGenerator(rescale=1./255)

train_generator = augment_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = datagen_test.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

# ==================== CLASS WEIGHT ====================
y_train = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# ==================== CALLBACKS ====================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join('logs', f'finetune_log_{timestamp}.csv')

csv_logger = CSVLogger(log_filename)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

callbacks = [csv_logger, early_stopping, reduce_lr]

# ==================== FINE-TUNE ====================
print("🚀 Starting fine-tuning...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# ==================== SAVE MODEL ====================
model.save('models/fineTunedModel.h5')
print("✅ Fine-tuned model saved to: models/fineTunedModel.h5")

# ==================== RENAME DAN EVALUASI ====================
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
acc_str = f"{final_acc:.4f}".replace(".", "_")
val_acc_str = f"{final_val_acc:.4f}".replace(".", "_")

final_model_renamed = f"models/fineTunedModel_val{val_acc_str}_acc{acc_str}.h5"
os.rename("models/fineTunedModel.h5", final_model_renamed)

print(f"\n✅ Final fine-tuned model renamed to: {final_model_renamed}")

# Optional: jika ingin auto-eval setelah fine-tune
print("🚀 Starting evaluation...")
import subprocess
subprocess.run([
    'python', '-m', 'src.evaluation',
    final_model_renamed,
    'datasets/raf-db-iterasi2/test',
    '64'
])