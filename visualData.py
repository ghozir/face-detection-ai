import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from glob import glob
import seaborn as sns

# --- Load CSV Log File ---
log_files = sorted(glob('logs/training_log_2025-04-23_18-37-07.csv'), key=os.path.getmtime, reverse=True)
latest_log_file = log_files[0] if log_files else None

if not latest_log_file:
    raise FileNotFoundError("❌ Tidak ditemukan file log di folder 'logs/'.")

df = pd.read_csv(latest_log_file)

# --- Plot Training Metrics ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Simpan plot ke file
plot_filename = f'logs/plotImage.png'
plt.savefig(plot_filename)
print(f"✅ Grafik disimpan ke: {plot_filename}")

# --- Training Summary ---
final_train_acc = df['accuracy'].iloc[-1]
final_val_acc = df['val_accuracy'].iloc[-1]
best_val_acc = df['val_accuracy'].max()
best_epoch = df['val_accuracy'].idxmax()

summary = {
    "Log File": os.path.basename(latest_log_file),
    "Final Train Accuracy": round(final_train_acc, 4),
    "Final Validation Accuracy": round(final_val_acc, 4),
    "Best Validation Accuracy": round(best_val_acc, 4),
    "Best Epoch": int(df['epoch'][best_epoch])
}

print("\n📊 Training Summary:")
print(pd.DataFrame([summary]).to_string(index=False))