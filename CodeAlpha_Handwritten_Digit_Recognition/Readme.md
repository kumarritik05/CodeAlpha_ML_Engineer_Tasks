# CodeAlpha – Handwritten Digit Recognition (Task 3)

## 📌 Internship Program
**Machine Learning Internship – CodeAlpha**

---

## 📖 Project Overview
This project implements a **Handwritten Digit Recognition System** using **Deep Learning**.  
The system accurately classifies handwritten digits (0–9) from grayscale images using a **Convolutional Neural Network (CNN)**.

Handwritten digit recognition is widely used in real-world applications such as:
- Optical Character Recognition (OCR)
- Bank cheque verification
- Postal code recognition
- Automated document processing

---

## 🎯 Objective
To build and train a machine learning model that can correctly identify handwritten digits from image data with high accuracy.

---

## 📊 Dataset
**MNIST Dataset**

- Total Images: **70,000**
- Image Size: **28 × 28 pixels**
- Image Type: **Grayscale**
- Classes: **Digits 0 to 9**

### Dataset Split
- Training Data: **60,000 images**
- Testing Data: **10,000 images**

The dataset is directly loaded using **TensorFlow/Keras**.

---

## 🧠 Model Description
**Convolutional Neural Network (CNN)**

### Architecture
- Convolution Layer (32 filters, ReLU activation)
- Max Pooling Layer
- Convolution Layer (64 filters, ReLU activation)
- Max Pooling Layer
- Flatten Layer
- Fully Connected Dense Layer (128 neurons)
- Dropout Layer (to prevent overfitting)
- Output Layer (10 neurons, Softmax activation)

---

## ⚙️ Methodology
1. Load the MNIST dataset
2. Normalize pixel values (0–255 → 0–1)
3. Reshape images for CNN input
4. One-hot encode labels
5. Build CNN architecture
6. Train the model using training data
7. Validate using test data
8. Predict handwritten digits

---

## 📈 Evaluation Metrics
- Accuracy
- Loss
- Validation Accuracy

---

## ✅ Results
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~98%
- The model successfully predicts handwritten digits from unseen images.

---

## 🛠️ Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## 📦 Requirements
All required dependencies are listed in `requirements.txt`.

```txt
tensorflow
numpy
matplotlib
scikit-learn
