import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess
from sklearn.utils import class_weight

from dualFolderSequence import DualFolderSequence  # pastikan file ini ada

# ==================== GPU CONFIG ====================
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

# ==================== PARAMETER ====================
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 100

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join('logs', f'training_log_{timestamp}.csv')

# ==================== DATASET COMBINED ====================
train_generator = DualFolderSequence(
    path1='datasets_augmented/train',
    path2='datasets/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    augment_params=None  # augmentasi sudah dilakukan offline
)

# Dataset validasi (tanpa augmentasi)
datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory(
    'datasets/test',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

steps_per_epoch = len(train_generator)
validation_steps = generator_test.samples // BATCH_SIZE

# ==================== CLASS WEIGHT ====================
y_train_labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_dict = dict(enumerate(class_weights))

print("\n📊 Calculated Class Weights:")
for class_idx, weight in class_weights_dict.items():
    class_name = list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(class_idx)]
    print(f"  - Class '{class_name}' (index {class_idx}): {weight:.2f}")
print("\n")

# ==================== RESIDUAL BLOCK ====================
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

# ==================== MODEL ARCHITECTURE ====================
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
outputs = tf.keras.layers.Dense(len(train_generator.class_indices),
                                activation='softmax',
                                name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet_StressClassifier')

# ==================== COMPILE ====================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ==================== CALLBACKS ====================
csv_logger = CSVLogger(log_filename)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
model_checkpoint_best = ModelCheckpoint('models/bestModel.h5', monitor='val_loss', save_best_only=True)
model_checkpoint_last = ModelCheckpoint('models/lastModel.h5', save_best_only=False)

callbacks = [csv_logger, early_stopping, reduce_lr, model_checkpoint_best, model_checkpoint_last]

# ==================== TRAINING ====================
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=generator_test,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# ==================== SAVE ====================
model.save('models/finalModel.h5')
print("\n✅ Training finished and model saved!")

print("🚀 Starting evaluation...")
subprocess.run([
    'python', '-m', 'src.evaluation',
    'models/finalModel.h5',
    'datasets/test',
    '64'
])
