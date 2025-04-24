import tensorflow as tf
import numpy as np
import cv2

img_size = 48

# Load model
model = tf.keras.models.load_model('final_model.h5')

# Label mapping (sesuaikan kalau perlu)
label_to_stress = {
    0: 'Stres Sangat Tinggi 😡',
    1: 'Stres Tinggi 😢',
    2: 'Stres Sedang 😯',
    3: 'Stres Rendah 😐',
    4: 'Stres Rendah 😊',
    5: 'N/A',
    6: 'N/A'
}

# Haar Cascade
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
        class_id = np.argmax(prediction)
        stress_text = label_to_stress.get(class_id, 'Unknown')

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{stress_text}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    cv2.imshow('Stress Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()