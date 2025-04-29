# 🧠 Face Detection AI with CNN

This project is a facial expression detection system based on **Convolutional Neural Network (CNN)** using TensorFlow.  
The model is trained to recognize facial emotions and provide insights into stress levels based on expressions.

---

## 📦 Initial Installation

1. **Clone this repository**
```bash
git clone https://github.com/ghozir/face-detection-ai.git
cd face-detection-ai
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
```

3. **Activate the virtual environment**
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Program

### 🔧 To Train the Model:
```bash
python3 trainingCnn.py
```

### 🧪 To Test the Model:
```bash
python3 modelTestMediapipe.py
```

### 📊 To Visualize Training Statistics (Accuracy & Loss):
```bash
python3 visualData.py
```

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `trainingCnn.py` | The main CNN model architecture for facial expression detection. |
| `modelTestMediapipe.py` | Evaluates the trained model on test data. |
| `visualData.py` | Displays and saves accuracy and loss graphs per epoch. |
| `requirements.txt` | List of required dependencies. |

---

## 🧠 Model Algorithm Overview

The model uses a CNN-based architecture with residual blocks to classify facial expressions. Here's a high-level summary of the training process:

1. **Data Preparation**:
   - Uses grayscale images resized to 48x48 pixels.
   - Combines original and augmented data using `ImageDataGenerator` to improve generalization.

2. **Model Architecture**:
   - Initial Conv2D layer with batch normalization and max pooling.
   - Four stacked **residual blocks** with increasing filter sizes: 64, 128, 256, 512.
   - Global average pooling followed by a dense layer and a softmax classifier.

3. **Training Configuration**:
   - Optimizer: Adam with `learning_rate=1e-4`
   - Loss function: Categorical Crossentropy with label smoothing
   - Callbacks used:
     - `EarlyStopping`: Stops training if `val_loss` doesn't improve after 10 epochs
     - `ReduceLROnPlateau`: Reduces learning rate if `val_loss` stagnates for 5 epochs
     - `ModelCheckpoint`: Saves the best model based on validation loss
     - `CSVLogger`: Logs training history to a timestamped CSV file

4. **Model Output**:
   - Trained model is saved as `models/finalModel.h5`
   - Best model (based on lowest validation loss) is saved as `models/bestModel.h5`

---

## 👨‍💼 Author

This program was created by **Ghozi Rabbani**  
Feel free to fork!

---

## 📜 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this code for personal or commercial purposes, as long as you give appropriate credit and include the original license.