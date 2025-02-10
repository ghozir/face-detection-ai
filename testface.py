import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import json

# Load model yang sudah dilatih
MODEL_PATH = "./app/models/stress_detection_model.keras"
LABELS_PATH = "./app/models/class_indices.json"
FER_TO_STRESS = {
    "angry": "stres_sangat_tinggi",
    "happy": "stres_rendah",
    "neutral": "stres_rendah",
    "sad": "stres_tinggi",
    "surprise": "stres_sedang"
}

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
    stress_labels = {v: k for k, v in class_indices.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Parameter gambar
IMG_WIDTH, IMG_HEIGHT = 48, 48

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)
        
        # Prediksi
        prediction = model.predict(face)
        predicted_class = np.argmax(prediction)
        predicted_label = stress_labels[predicted_class]
        stress_level = FER_TO_STRESS[predicted_label]
        
        # Tampilkan hasil prediksi
        cv2.putText(frame, f"Stress Level: {stress_level}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Tampilkan frame
    cv2.imshow("Stress Detection", frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
