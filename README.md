# 🧠 Stress Level Classifier with CNN

This project is a stress level detection system based on **Convolutional Neural Network (CNN)** using TensorFlow.  
The model is trained to recognize facial expressions and map them into different stress levels.

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
(Windows: `venv\Scripts\activate`)

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Program

### 🔧 To Train the Model:
```bash
python3 -m src.training
```
*(Training automatically triggers evaluation after completion.)*

### 🧪 To Test the Model:
- Using OpenCV only:
```bash
python3 -m src.modelTest
```
- Using Mediapipe:
```bash
python3 -m src.modelTestMediapipe
```

### 📊 To Visualize Training Statistics (Accuracy & Loss):
```bash
python3 -m src.visualData
```

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `src/training.py` | Main training script for CNN model and automatic evaluation after training. |
| `src/evaluation.py` | Evaluate saved models and generate confusion matrix + classification report. |
| `src/modelTest.py` | Test the trained model in real-time using OpenCV only. |
| `src/modelTestMediapipe.py` | Test the trained model in real-time using Mediapipe face detection. |
| `src/visualData.py` | Visualize and save training history (accuracy and loss per epoch). |
| `requirements.txt` | List of required dependencies. |

---

## 🧠 Model Algorithm Overview

The model uses a CNN-based architecture enhanced with residual blocks to classify facial expressions. Here's a high-level summary:

1. **Data Preparation**:
   - Grayscale images resized to **48x48** or **64x64** pixels (configurable).
   - Combines original and augmented data using `ImageDataGenerator` for improved generalization.

2. **Model Architecture**:
   - Initial Conv2D layer with batch normalization and max pooling.
   - Four stacked **residual blocks** with increasing filter sizes: 64, 128, 256, 512.
   - Global average pooling followed by dense layers, LeakyReLU activation, and softmax output.

3. **Training Configuration**:
   - Optimizer: Adam (`learning_rate=1e-4`)
   - Loss function: Categorical Crossentropy with label smoothing
   - Callbacks used:
     - `EarlyStopping`: Stops training if validation loss doesn't improve for 10 epochs
     - `ReduceLROnPlateau`: Reduces learning rate if validation loss plateaus for 5 epochs
     - `ModelCheckpoint`: Saves the best model based on validation loss
     - `CSVLogger`: Logs training history into timestamped CSV files

4. **Evaluation & Results**:
   - After training, the model is immediately evaluated.
   - Confusion matrix and classification report are automatically generated.
   - Outputs are saved inside the `logs/` folder.

5. **Model Output**:
   - Last trained model: `models/finalModel.h5`
   - Best validation model: `models/bestModel.h5`

---

## 👨‍💼 Author

Created by **Ghozi Rabbani**  
Feel free to fork, star, or contribute!

---

## 📜 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this code for personal or commercial purposes, as long as you give proper credit and include the original license.