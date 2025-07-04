from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Create directories if they don't exist
os.makedirs('static/sample_images', exist_ok=True)

class DigitRecognizer:
    def __init__(self, model_path='models/mnist_cnn.h5'):
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)
    
    def preprocess_image(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, img):
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img)
        return np.argmax(prediction), np.max(prediction)

recognizer = DigitRecognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = np.array(img)
    
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    digit, confidence = recognizer.predict(img)
    
    return jsonify({
        'digit': int(digit),
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)