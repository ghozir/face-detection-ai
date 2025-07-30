
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import subprocess
from sklearn.utils import class_weight

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

# ==================== DATASET ====================
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
    'datasets/raf-db/train',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

generator_test = datagen_test.flow_from_directory(
    'datasets/raf-db/test',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

steps_per_epoch = train_generator.samples // BATCH_SIZE
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
outputs = tf.keras.layers.Dense(train_generator.num_classes,
                                activation='softmax',
                                name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet_StressClassifier')

# ==================== COMPILE ====================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ==================== CUSTOM CALLBACK FOR NAMED SAVING ====================
class SaveModelWithMetrics(tf.keras.callbacks.Callback):
    def __init__(self, prefix='bestModel', save_best=True, monitor='val_loss', mode='min'):
        super().__init__()
        self.prefix = prefix
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_value = np.Inf if mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        metric = logs.get(self.monitor, None)
        if metric is None:
            return

        if self.save_best:
            val_acc = logs.get('val_accuracy', 0.0)
            train_acc = logs.get('accuracy', 0.0)

            save_condition = (
                (self.mode == 'min' and metric < self.best_value) or
                (self.mode == 'max' and metric > self.best_value)
            )

            if save_condition:
                self.best_value = metric
                acc_str = f"{train_acc:.4f}".replace('.', '_')
                val_acc_str = f"{val_acc:.4f}".replace('.', '_')
                filename = f"models/{self.prefix}_val{val_acc_str}_acc{acc_str}.h5"
                self.model.save(filename)
                print(f"📁 Saved {self.prefix} model at: {filename}")
        else:
            # Simpan model setiap epoch dengan nama tetap (overwrite)
            filename = f"models/{self.prefix}.h5"
            self.model.save(filename)
            print(f"💾 Saved last model checkpoint: {filename}")

# ==================== CALLBACKS ====================
csv_logger = CSVLogger(log_filename)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

save_best = SaveModelWithMetrics(prefix='bestModel', save_best=True, monitor='val_loss', mode='min')
save_last = SaveModelWithMetrics(prefix='lastModel', save_best=False)

callbacks = [csv_logger, early_stopping, reduce_lr, save_best, save_last]

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
# ==================== SAVE FINAL MODEL WITH METRICS ====================
model.save('models/finalModel.h5')

final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
acc_str = f"{final_acc:.4f}".replace(".", "_")
val_acc_str = f"{final_val_acc:.4f}".replace(".", "_")

final_model_renamed = f"models/finalModel_val{val_acc_str}_acc{acc_str}.h5"
os.rename("models/finalModel.h5", final_model_renamed)

print(f"\n✅ Final model renamed to: {final_model_renamed}")

print("🚀 Starting evaluation...")
subprocess.run([
    'python', '-m', 'src.evaluation',
    final_model_renamed,
    'datasets/raf-db/test',
    '64'
])