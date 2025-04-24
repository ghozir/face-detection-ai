import cv2
import numpy as np
import tensorflow as tf

# Load model yang sudah kamu training
model = tf.keras.models.load_model('best_model.h5')

# Kelas emosi (urutan harus sesuai urutan label dari dataset kamu ya)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map ekspresi ke estimasi stress level
stress_map = {
    'Happy': 'Stres Rendah (0-20%)',
    'Neutral': 'Stres Rendah (20-35%)',
    'Surprise': 'Stres Sedang (35-60%)',
    'Sad': 'Stres Tinggi (60-80%)',
    'Angry': 'Stres Sangat Tinggi (80-100%)',
    'Disgust': 'Ekspresi Tidak Digunakan',
    'Fear': '??? (opsional, bisa kamu skip kalau ga relevan)'
}

# Load Haar Cascade buat deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=(0, -1))

        prediction = model.predict(face_resized)
        pred_label_idx = np.argmax(prediction)
        label = class_labels[pred_label_idx]
        stress_level = stress_map.get(label, "Unknown")

        # Draw rectangle dan text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{label}: {stress_level}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Stress Detector by Ekspresi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
