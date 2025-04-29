'''
Stress Level Classifier Training Script (Updated)
- ResNet-style CNN model
- Data Augmentation
- GPU Config
- Callback: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
- Fix warning keras (LeakyReLU, Layer naming)
'''

import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ====================
# GPU Detection & Configuration
# ====================
print("✅ GPU devices:", tf.config.list_physical_devices('GPU'))
print("🔥 GPU name:", tf.test.gpu_device_name())

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("🚀 GPU detected, enabling memory growth...")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"⚠️ Failed to set memory growth: {e}")
else:
    print("💻 No GPU detected, training on CPU...")

# ====================
# Parameters
# ====================
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 100

# Create logs directory
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join('logs', f'training_log_{timestamp}.csv')

# ====================
# Data Generators
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

# Combine generators

def combined_generator(gen1, gen2):
    while True:
        imgs1, labels1 = next(gen1)
        imgs2, labels2 = next(gen2)
        yield np.concatenate((imgs1, imgs2), axis=0), np.concatenate((labels1, labels2), axis=0)

train_generator = combined_generator(generator_train_orig, generator_train_aug)
steps_per_epoch = (generator_train_orig.samples + generator_train_aug.samples) // BATCH_SIZE
validation_steps = generator_test.samples // BATCH_SIZE

# ====================
# Residual Block
# ====================

def residual_block(x, filters, block_num):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
                                          kernel_regularizer=l2(1e-4),
                                          name=f'shortcut_conv_block{block_num}')(shortcut)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same',
                               kernel_regularizer=l2(1e-4),
                               name=f'conv1_block{block_num}')(x)
    x = tf.keras.layers.BatchNormalization(name=f'bn1_block{block_num}')(x)
    x = tf.keras.layers.ReLU(name=f'relu1_block{block_num}')(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same',
                               kernel_regularizer=l2(1e-4),
                               name=f'conv2_block{block_num}')(x)
    x = tf.keras.layers.BatchNormalization(name=f'bn2_block{block_num}')(x)

    x = tf.keras.layers.add([shortcut, x], name=f'add_block{block_num}')
    x = tf.keras.layers.ReLU(name=f'relu_out_block{block_num}')(x)
    return x

# ====================
# Model Architecture
# ====================

inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name='input_image')

x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                           kernel_regularizer=l2(1e-4), name='conv1_initial')(inputs)
x = tf.keras.layers.BatchNormalization(name='bn_conv1_initial')(x)
x = tf.keras.layers.ReLU(name='relu_conv1_initial')(x)
x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)

x = residual_block(x, 64, block_num=1)
x = residual_block(x, 128, block_num=2)
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
x = residual_block(x, 256, block_num=3)
x = residual_block(x, 512, block_num=4)

x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dense(128, kernel_regularizer=l2(1e-4), name='fc1')(x)
x = tf.keras.layers.LeakyReLU(negative_slope=0.1, name='leaky_relu')(x)
x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
outputs = tf.keras.layers.Dense(generator_train_orig.num_classes,
                                activation='softmax',
                                name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet_StressClassifier')

# ====================
# Compile Model
# ====================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ====================
# Callbacks
# ====================
csv_logger = CSVLogger(log_filename)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
model_checkpoint_best = ModelCheckpoint('models/bestModel.h5', monitor='val_loss', save_best_only=True)
model_checkpoint_last = ModelCheckpoint('models/lastModel.h5', save_best_only=False)

callbacks = [csv_logger, early_stopping, reduce_lr, model_checkpoint_best, model_checkpoint_last]

# ====================
# Train Model
# ====================
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=generator_test,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save('models/finalModel.h5')

print("\n✅ Training finished and model saved!")