import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Load dataset FER-2013
data_dir = 'path_to_fer2013_dataset'  # Ubah dengan path dataset FER-2013
img_size = 48
batch_size = 64

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size, color_mode='grayscale', class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size, color_mode='grayscale', class_mode='categorical', subset='validation')

# Build Residual Block
def residual_block(x, filters):
    res = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    res = BatchNormalization()(res)
    res = Conv2D(filters, (3,3), padding='same', activation='relu')(res)
    res = BatchNormalization()(res)
    x = Add()([x, res])
    return x

# Build Hybrid Model (Residual Network + CNN)
inputs = Input(shape=(img_size, img_size, 1))
x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = residual_block(x, 64)
x = residual_block(x, 128)
x = residual_block(x, 256)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(7, activation='softmax')(x) # 7 kelas emosi FER-2013

hybrid_model = Model(inputs, x)

hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hybrid_model.summary()

# Train Hybrid Model
hybrid_model.fit(train_data, validation_data=val_data, epochs=30)

# Save model
hybrid_model.save('stress_level_hybrid_model.h5')

# Real-time Stress Detection with OpenCV
def detect_stress():
    model = tf.keras.models.load_model('stress_level_hybrid_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (img_size, img_size))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = face / 255.0
            
            prediction = model.predict(face)
            stress_score = np.argmax(prediction) * (100 / 6)  # Convert to 1-100 range
            stress_level = 'High' if stress_score >= 75 else 'Low'
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Stress: {stress_level} ({int(stress_score)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        cv2.imshow('Stress Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
detect_stress()