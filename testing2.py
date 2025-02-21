import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Load dataset FER-2013
data_dir = './dataset'  # Ubah dengan path dataset FER-2013
img_size = 48
batch_size = 64

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size, color_mode='grayscale', class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size, color_mode='grayscale', class_mode='categorical', subset='validation')

# Build CNN Model
cnn_model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax') # 7 kelas emosi FER-2013
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Train Model
cnn_model.fit(train_data, validation_data=val_data, epochs=30)

# Save model
cnn_model.save('stress_level_cnn_model.h5')
