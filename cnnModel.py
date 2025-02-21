import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

# Parameter model
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50

# Data Augmentation yang lebih bervariasi
train_datagen = ImageDataGenerator(
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

# Residual Block dengan L2 Regularization
def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(shortcut)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Model CNN dengan Residual Blocks
inputs = tf.keras.layers.Input(shape=(48, 48, 1))
x = tf.keras.layers.BatchNormalization()(inputs)  # BatchNorm diawal

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = residual_block(x, 32)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = residual_block(x, 64)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = residual_block(x, 128)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = residual_block(x, 256)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(generator_train.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs, outputs)

# Optimizer AdamW
opt = AdamW(learning_rate=0.001, weight_decay=1e-5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

# Training model
model.fit(generator_train, 
          validation_data=generator_test, 
          epochs=EPOCHS, 
          callbacks=[early_stopping, lr_scheduler, checkpoint])

# Simpan model
model.save('stress_detection_model_final.h5')
print("Model berhasil disimpan sebagai 'stress_detection_model_final.h5'")
