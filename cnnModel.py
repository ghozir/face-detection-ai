import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Parameter model
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 30

# Augmentasi data yang kompleks
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
    'dataset/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

generator_test = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 🔥 Residual Block dengan Penyesuaian Shortcut jika Diperlukan
def residual_block(x, filters):
    shortcut = x  # Simpan shortcut asli

    # Jika jumlah filter berbeda, gunakan Conv2D(1x1) agar ukuran cocok
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    # Convolusi pertama
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Convolusi kedua tanpa aktivasi dulu
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tambahkan shortcut untuk residual connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

# 🔥 Model CNN + Residual Block
inputs = tf.keras.layers.Input(shape=(48, 48, 1))

# Conv awal
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)

# 🔹 Tambahkan Residual Block
x = residual_block(x, 32)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# 🔹 Residual Block lagi dengan filter meningkat
x = residual_block(x, 64)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# 🔹 Residual Block lagi dengan filter meningkat
x = residual_block(x, 128)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Fully Connected Layer
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Output Layer (7 kelas emosi)
outputs = tf.keras.layers.Dense(generator_train.num_classes, activation='softmax')(x)

# Bangun model
model = tf.keras.models.Model(inputs, outputs)

# Optimizer Adam dengan learning rate rendah
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# EarlyStopping untuk menghindari overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Training model
model.fit(generator_train, 
          validation_data=generator_test, 
          epochs=EPOCHS, 
          callbacks=[early_stopping], 
          workers=4,  # Atur jumlah worker untuk parallel processing
          use_multiprocessing=True)  # Aktifkan multiprocessing

# Simpan model dalam format .h5
model.save('stress_detection_model.h5')

print("Model berhasil disimpan sebagai 'stress_detection_model.h5'")
