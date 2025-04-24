import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Load dataset FER-2013
img_size = 48
batch_size = 64


# Real-time Stress Detection with OpenCV
def detect_stress():
    model = tf.keras.models.load_model('final_model.h5')
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