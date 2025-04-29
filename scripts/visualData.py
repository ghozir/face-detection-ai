"""
Training Log Plotter and Summary

This script loads the most recent training log CSV from the 'logs/' directory,
plots accuracy and loss curves over epochs, saves the resulting figure,
and prints a summary of final and best validation metrics.
"""

import os
from glob import glob
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# ====================
# Locate Latest Log File
# ====================
"""
Finds the most recent CSV log file in the 'logs/' folder by modification time.
Raises FileNotFoundError if no logs are present.
"""
log_folder = 'logs'
csv_pattern = os.path.join(log_folder, 'training_log_2025-04-25_23-51-17.csv') # Replace with your log file pattern
log_files = sorted(glob(csv_pattern), key=os.path.getmtime, reverse=True)
if not log_files:
    raise FileNotFoundError(f"❌ No log files found in '{log_folder}/'.")
latest_log_file = log_files[0]
print(f"✅ Loading log file: {latest_log_file}")

# ====================
# Load CSV into DataFrame
# ====================
"""
Reads the training log CSV into a pandas DataFrame for plotting.
"""
df = pd.read_csv(latest_log_file)

# ====================
# Plot Training Metrics
# ====================
"""
Plots training and validation accuracy and loss over each epoch,
and saves the figure to 'logs/plotImage.png'.
"""
plt.figure(figsize=(14, 6))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot image
plot_filename = os.path.join(log_folder, f'plotResult.png')
plt.savefig(plot_filename)
print(f"✅ Plot saved to: {plot_filename}")

# ====================
# Training Summary
# ====================
"""
Computes and prints a summary including final and best validation accuracy and the epoch at which it occurred.
"""
final_train_acc = df['accuracy'].iloc[-1]
final_val_acc = df['val_accuracy'].iloc[-1]
best_val_acc = df['val_accuracy'].max()
best_epoch = int(df.loc[df['val_accuracy'].idxmax(), 'epoch'])

summary = pd.DataFrame([{
    'Log File': os.path.basename(latest_log_file),
    'Final Train Accuracy': round(final_train_acc, 4),
    'Final Validation Accuracy': round(final_val_acc, 4),
    'Best Validation Accuracy': round(best_val_acc, 4),
    'Best Epoch': best_epoch
}])

print("\n📊 Training Summary:")
print(summary.to_string(index=False))