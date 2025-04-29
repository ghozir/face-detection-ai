import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

# Parameter model
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 100

# Data Augmentation
augment_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

# Data tanpa augmentasi
original_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset (tanpa augmentasi)
generator_train_original = original_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE // 2,
    class_mode='categorical'
)

# Load dataset (dengan augmentasi)
generator_train_augmented = augment_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE // 2,
    class_mode='categorical'
)

# Load dataset untuk testing/validation
generator_test = original_datagen.flow_from_directory(
    'dataset/test',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Custom generator untuk menggabungkan data asli dan augmentasi
def combined_generator(gen1, gen2):
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        yield (np.concatenate((batch1[0], batch2[0])), np.concatenate((batch1[1], batch2[1])))

train_generator = combined_generator(generator_train_original, generator_train_augmented)

# Menghitung jumlah langkah (steps per epoch)
steps_per_epoch = (generator_train_original.samples + generator_train_augmented.samples) // BATCH_SIZE
validation_steps = generator_test.samples // BATCH_SIZE

# Residual Block dengan L2 Regularization
def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(shortcut)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.add([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Membuat model CNN
inputs = tf.keras.Input(shape=(*IMG_SIZE, 1))  # Karena grayscale (1 channel)
x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', kernel_regularizer=l2(1e-4))(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

x = residual_block(x, 64)
x = residual_block(x, 128)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = residual_block(x, 256)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(generator_train_original.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Setup callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=generator_test,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Setelah training, kamu bisa save modelnya lagi kalau mau
model.save('final_model.h5')
