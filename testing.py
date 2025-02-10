import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Parameter model
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50  # Perpanjang jumlah epoch

# Augmentasi data
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

# Model CNN dengan arsitektur yang lebih kompleks dan BatchNormalization
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(generator_train.num_classes, activation='softmax')
])

# Optimizer dengan learning rate yang lebih rendah
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Custom Data Generator untuk Augmentasi Tambahan
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
    
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        images, labels = self.generator[index]
        augmented_images = np.array([train_datagen.random_transform(image) for image in images])
        images = np.concatenate([images, augmented_images], axis=0)
        labels = np.concatenate([labels, labels], axis=0)
        
        # Pastikan format sesuai dengan yang dibutuhkan model
        images = np.expand_dims(images, axis=-1) if images.shape[-1] != 1 else images
        
        return images, labels

generator_train = CustomDataGenerator(generator_train)
generator_test = CustomDataGenerator(generator_test)

# EarlyStopping untuk menghindari overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)


# Training dengan EarlyStopping
print(f"Epochs yang digunakan: {EPOCHS}")

model.fit(generator_train, validation_data=generator_test, epochs=EPOCHS, callbacks=[early_stopping])
