"""
Real-time Stress Detection Inference Script

Loads a pretrained CNN model for stress classification, initializes MediaPipe face detection,
and performs real-time inference on webcam frames. Annotates detected faces with emotion
labels and corresponding stress levels, displaying results in a live video window.
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ====================
# Model Loading & Inference Setup
# ====================
"""
Load the saved Keras model and wrap it in a tf.function for optimized, stateless inference.
Also perform a dummy inference to allocate GPU/CPU resources ahead of time.
"""
model = tf.keras.models.load_model('models/finalModel.h5')
infer = tf.function(lambda x: model(x, training=False))
# Warm-up inference to avoid first-call latency
_ = infer(tf.zeros((1, 48, 48, 1), dtype=tf.float32))

# ====================
# Label Definitions & Stress Mapping
# ====================
"""
Define the mapping from model output indices to emotion labels and then to stress levels.
"""
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
stress_map = {
    'happy': 'Stres Rendah',
    'neutral': 'Stres Rendah',
    'surprise': 'Stres Sedang',
    'sad': 'Stres Sedang',
    'angry': 'Stres Tinggi',
    'fear': 'Stres Sangat Tinggi',
    'disgust': 'Stres Sedang'
}

# ====================
# MediaPipe Face Detection Initialization
# ====================
"""
Set up MediaPipe Face Detection with a model selection optimized for lower resolution frames
and a minimum detection confidence threshold.
"""
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(
    model_selection=1,             # Use lightweight model for <=1280×720 frames
    min_detection_confidence=0.5   # Minimum confidence for detections
)

# ====================
# Video Capture Setup
# ====================
"""
Open the default webcam (device 0) and define a target width for faster frame processing.
"""
cap = cv2.VideoCapture(0)
scale_width = 640

# ====================
# Main Processing Loop
# ====================
"""
Continuously read frames from the webcam, detect faces, crop and preprocess each face
for the classifier, run inference, then draw bounding boxes and labels on the original frame.
Press 'q' to exit the loop.
"""
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame not captured

    # Resize frame for faster face detection
    h, w = frame.shape[:2]
    scale_factor = scale_width / float(w)
    frame_small = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    # Convert BGR to RGB for MediaPipe processing
    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detector.process(rgb_small)
    if results.detections:
        for detection in results.detections:
            # Get normalized bounding box and convert to pixel coordinates
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * frame_small.shape[1])
            y1 = int(bbox.ymin * frame_small.shape[0])
            w1 = int(bbox.width * frame_small.shape[1])
            h1 = int(bbox.height * frame_small.shape[0])

            # Map coordinates back to original frame size
            x_orig = int(x1 / scale_factor)
            y_orig = int(y1 / scale_factor)
            w_orig = int(w1 / scale_factor)
            h_orig = int(h1 / scale_factor)

            # Crop face region and convert to grayscale
            face = frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Resize to model input size and normalize pixel values
            resized = cv2.resize(gray, (48, 48))
            inp = resized.astype('float32') / 255.0
            inp = np.expand_dims(inp, axis=(0, -1))  # Shape: (1, 48, 48, 1)

            # Run inference and determine predicted label and stress level
            preds = infer(inp).numpy()
            idx = np.argmax(preds)
            label = class_labels[idx]
            stress = stress_map.get(label, 'Unknown')

            # Draw bounding box and annotation text on the original frame
            cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{label} | {stress}",
                (x_orig, y_orig - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display annotated frame in a window
    cv2.imshow('Stress Detector (MediaPipe)', frame)
    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====================
# Cleanup Resources
# ====================
"""
Release the video capture device and close all OpenCV display windows.
"""
cap.release()
cv2.destroyAllWindows()