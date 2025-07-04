# 🧠 Handwritten Digit Recognizer

A deep learning web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. This project includes a Flask-powered web interface where users can draw digits and receive instant predictions.



## 🚀 Features

- 🧠 Deep learning CNN model (trained on MNIST)
- 🖼️ Draw digits directly in the browser
- 🔮 Real-time digit prediction with confidence score
- 📈 Training accuracy/loss visualization
- 🧪 Command-line prediction using saved model
- 🔗 RESTful API endpoint (`/predict`)

---

## 🗂️ Project Structure

handwritten-digit-recognizer/
│
├── models/ # Saved Keras model
│ └── mnist_cnn.h5
│
├── scripts/ # Training and prediction scripts
│ ├── train.py
│ └── predict.py
│
├── static/ # Static files
│ ├── sample_images/ # Example digit images
│ └── styles.css
│
├── templates/
│ └── index.html # Web interface
│
├── app.py # Flask web server
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── LICENSE # MIT License
└── .gitignore # Git configuration

yaml
Copy
Edit

---

## 🧑‍💻 Installation

### 1. Clone the repository
```bash
git clone https://github.com/bijayakumar2005/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
2. Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies


pip install -r requirements.txt
🏋️‍♂️ Train the Model (Optional)
You can train your own model using the MNIST dataset:


python scripts/train.py
This will save mnist_cnn.h5 inside the models/ directory.

🔮 Run Prediction from Image (Optional)
To test prediction using a sample image:


python scripts/predict.py static/sample_images/digit7.png
🌐 Run the Web App
Start the Flask app with:


python app.py
Visit http://127.0.0.1:5000 in your browser.

🧪 Sample Images
If you need test images for prediction, use the ones in static/sample_images/, or generate new ones in any 28x28 format (black background, white digit).



📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
MNIST Dataset

TensorFlow + Keras

Flask framework

Author : Bijaya kumar rout
Email : bijayakumarrout2005@gmail.com
