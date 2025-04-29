import cv2
import numpy as np
import tensorflow as tf

# Load model dan wrap inferensi dengan tf.function untuk kinerja lebih baik
model = tf.keras.models.load_model('finalModel.h5')
infer = tf.function(lambda x: model(x, training=False))

# Warm-up untuk menghindari lag pada deteksi pertama
_ = infer(tf.zeros((1, 48, 48, 1), dtype=tf.float32))

# Kelas emosi dan mapping ke level stres
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
stress_map = {
    'happy':    'Stres Rendah',
    'neutral':  'Stres Rendah',
    'surprise': 'Stres Sedang',
    'sad':      'Stres Sedang',
    'angry':    'Stres Tinggi',
    'fear':     'Stres Sangat Tinggi',
    'disgust':  'Stres Sedang'
}

# Inisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame ke lebar 640 untuk mempercepat deteksi
    scale_width = 640
    h, w = frame.shape[:2]
    scaling_factor = scale_width / float(w)
    frame_small = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah di frame kecil
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, fw, fh) in faces:
        # Kembalikan koordinat ke frame asli
        x_orig = int(x / scaling_factor)
        y_orig = int(y / scaling_factor)
        w_orig = int(fw / scaling_factor)
        h_orig = int(fh / scaling_factor)

        # Crop & preprocess
        face = cv2.cvtColor(frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig], cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face, (48, 48))
        face_norm = face_resized.astype('float32') / 255.0
        input_tensor = np.expand_dims(face_norm, axis=(0, -1))  # shape (1,48,48,1)

        # Inferensi cepat via tf.function tanpa overhead predict()
        preds = infer(input_tensor).numpy()
        idx = np.argmax(preds)
        label = class_labels[idx]
        stress = stress_map.get(label, 'Unknown')

        # Gambar hasil
        cv2.rectangle(frame, (x_orig, y_orig), (x_orig+w_orig, y_orig+h_orig), (255, 0, 0), 2)
        cv2.putText(frame, f'{label} | {stress}', (x_orig, y_orig-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Real-Time Stress Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
