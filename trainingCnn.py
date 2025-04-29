"""
Stress Level Classifier Training Script

This script defines, trains, and saves a ResNet-style CNN model to classify stress levels
based on facial expression images. It includes GPU configuration, data loading with
augmentation, model architecture, compilation, callbacks, and training routines.
"""

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import matplotlib.pyplot as plt

# ====================
# GPU Detection & Configuration
# ====================
print("✅ GPU devices:", tf.config.list_physical_devices('GPU'))  # List available GPU devices
print("🔥 GPU name:", tf.test.gpu_device_name())               # Print the name of the GPU in use, if any

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("🚀 GPU detected, enabling GPU for training...")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"⚠️ Failed to set memory growth: {e}")
else:
    print("💻 No GPU detected, proceeding with CPU...")

# ====================
# Training Parameters
# ====================
IMG_SIZE = (48, 48)   # Input image dimensions (width, height)
BATCH_SIZE = 32       # Number of samples per gradient update
EPOCHS = 100          # Maximum number of training epochs

# Create directory for logs if it doesn't exist
directory = 'logs'
os.makedirs(directory, exist_ok=True)

# Generate log filename with current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(directory, f'training_log_{timestamp}.csv')

# ====================
# Data Augmentation & Generators
# ====================
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

# Generator for validation and unaugmented training data
datagen_no_aug = ImageDataGenerator(rescale=1./255)

generator_train_orig = datagen_no_aug.flow_from_directory(
    'datasets/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE // 2,
    class_mode='categorical'
)

generator_train_aug = augment_datagen.flow_from_directory(
    'datasets/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE // 2,
    class_mode='categorical'
)

generator_test = datagen_no_aug.flow_from_directory(
    'datasets/test',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

def combined_generator(gen1, gen2):
    """
    Merge batches from two generators into a single batch.

    Args:
        gen1: A Keras ImageDataGenerator iterator for original data.
        gen2: A Keras ImageDataGenerator iterator for augmented data.

    Yields:
        Tuple of (images, labels) concatenated from both generators.
    """
    while True:
        imgs1, labels1 = next(gen1)
        imgs2, labels2 = next(gen2)
        combined_imgs = np.concatenate((imgs1, imgs2), axis=0)
        combined_labels = np.concatenate((labels1, labels2), axis=0)
        yield combined_imgs, combined_labels

train_generator = combined_generator(generator_train_orig, generator_train_aug)

# Compute steps for training and validation
steps_per_epoch = (generator_train_orig.samples + generator_train_aug.samples) // BATCH_SIZE
validation_steps = generator_test.samples // BATCH_SIZE

# ====================
# Residual Block Definition
# ====================
def residual_block(x, filters):
    """
    Build a ResNet-style residual block.

    Args:
        x: Input tensor.
        filters: Number of filters for convolutions.

    Returns:
        Output tensor after applying two convolutional layers and skip connection.
    """
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
                                          kernel_regularizer=l2(1e-4),
                                          name='shortcut_conv')(shortcut)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same',
                               kernel_regularizer=l2(1e-4),
                               name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same',
                               kernel_regularizer=l2(1e-4),
                               name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.add([shortcut, x], name='add')
    x = tf.keras.layers.ReLU(name='relu_out')(x)
    return x

# ====================
# Model Architecture
# ====================
inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name='input_image')
"""
Define the model input layer accepting grayscale images of size IMG_SIZE.
"""

# Initial convolution and pooling layers
x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                           kernel_regularizer=l2(1e-4), name='conv1')(inputs)
x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x)
x = tf.keras.layers.ReLU(name='relu_conv1')(x)
x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)

# Stack residual blocks with increasing filter sizes
x = residual_block(x, 64)
x = residual_block(x, 128)
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
x = residual_block(x, 256)
x = residual_block(x, 512)

# Global pooling and fully connected classification head
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dense(128, kernel_regularizer=l2(1e-4), name='fc1')(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu')(x)
x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
outputs = tf.keras.layers.Dense(generator_train_orig.num_classes,
                                activation='softmax',
                                name='predictions')(x)

# Instantiate the Keras Model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet_StressClassifier')

# ====================
# Compile Model
# ====================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
"""
Compile the model with Adam optimizer, categorical crossentropy loss with label smoothing,
and track accuracy metric.
"""

# ====================
# Callbacks Setup
# ====================
# CSVLogger: Write training metrics to CSV file
csv_logger = CSVLogger(log_filename, append=False)
# EarlyStopping: Stop training if validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)
# ReduceLROnPlateau: Halve learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
# ModelCheckpoint: Save model weights of the best performing epoch
model_checkpoint = ModelCheckpoint('models/bestModel.h5',
                                   monitor='val_loss',
                                   save_best_only=True)
callbacks = [csv_logger, early_stopping, reduce_lr, model_checkpoint]

# ====================
# Model Training
# ====================
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=generator_test,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)
"""
Train the model using the combined training generator and validate on the test set.
History object contains loss and accuracy metrics over epochs.
"""

# Save the final trained model to disk
model.save('models/finalModel.h5')
"""
Persist the trained model for future inference or fine-tuning.
"""