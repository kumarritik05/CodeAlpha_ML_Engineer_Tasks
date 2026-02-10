import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("model/digit_model.h5")

# Load MNIST test data
(_, _), (X_test, y_test) = mnist.load_data()

# Normalize
X_test = X_test / 255.0

# Select an image
index = 7
image = X_test[index]

# Reshape for prediction
image_input = image.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(image_input)
predicted_digit = np.argmax(prediction)

# Save prediction image
plt.imshow(image, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.savefig("outputs/sample_prediction.png")
plt.close()

print(f"✅ Predicted Digit: {predicted_digit}")
print("✅ Prediction image saved in outputs folder")
