import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import argparse
import matplotlib.pyplot as plt

class DigitRecognizer:
    def __init__(self, model_path='models/mnist_cnn.h5'):
        self.model = keras.models.load_model(model_path)
    
    def preprocess_image(self, img):
        # Convert to grayscale if needed
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert colors (MNIST has white digits on black background)
        img = cv2.bitwise_not(img)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize and reshape for model input
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        
        return img
    
    def predict(self, img):
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_digit, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict handwritten digit from image file.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    recognizer = DigitRecognizer()
    
    # Load and predict
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        exit(1)
    
    digit, confidence = recognizer.predict(img)
    print(f"Predicted Digit: {digit} with confidence: {confidence:.2%}")
    
    # Display the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {digit} ({confidence:.2%})")
    plt.axis('off')
    plt.show()