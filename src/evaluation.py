# evaluation.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import sys
import os
from datetime import datetime

def evaluate_model(model_path, test_dir, img_size, batch_size, timestamp):
    print("🔍 Loading model from:", model_path)
    model = tf.keras.models.load_model(model_path)

    print("📁 Loading test dataset from:", test_dir)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    x_test, y_test = [], []
    print("📦 Preparing test data for evaluation...")
    for i in range(len(generator)):
        imgs, labels = generator[i]
        x_test.append(imgs)
        y_test.append(labels)
        if (i + 1) * batch_size >= generator.samples:
            break

    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    print("🤖 Predicting test data...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    class_names = list(generator.class_indices.keys())

    print("📊 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    cm_path = f'logs/confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path)
    plt.show()

    print(f"✅ Confusion matrix saved to {cm_path}")

    print("\n📄 Generating classification report...")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Print ke terminal
    print(report_df)

    report_path = f'logs/classification_report_{timestamp}.csv'
    report_df.to_csv(report_path)

    print(f"✅ Classification report saved to {report_path}")

# Kalau script ini dijalankan langsung, ambil argumen CLI
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("❌ Usage: python evaluation.py <model_path> <test_dir> [timestamp]")
        sys.exit(1)

    model_path = sys.argv[1]
    test_dir = sys.argv[2]
    img_size_val = int(sys.argv[3])  # 🔥 Ambil img_size dari argumen CLI
    img_size = (img_size_val, img_size_val)  # 🔥 Bentuk tuple (width, height)
    timestamp = sys.argv[4] if len(sys.argv) == 5 else datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # ✅ PANGGIL FUNGSI EVALUASI
    evaluate_model(
        model_path=model_path,
        test_dir=test_dir,
        img_size=img_size,
        batch_size=32,
        timestamp=timestamp
    )