from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)
model = load_model('trained.h5')  # Load your trained model

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/database'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

def is_xray(img):
    """Check if the image resembles an X-ray"""
    if len(img.shape) == 3 and img.shape[2] == 3:  # If RGB, convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # X-rays typically have high contrast (dark background with bright regions)
    mean_pixel = np.mean(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Adjust these thresholds based on your X-ray dataset
    if mean_pixel < 30 or mean_pixel > 220:
        return False
    return True

def predict_pneumonia(file_path):
    """Predict pneumonia from X-ray image"""
    img = cv2.imread(file_path)
    if img is None:
        return "Error: Could not read image"
    
    # Verify it's an X-ray
    if not is_xray(img):
        return "Error: Not a valid X-ray image"
    
    # Preprocess for model
    img = cv2.resize(img, (300, 300))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 300, 300, 3)  # Reshape for model input
    
    try:
        prediction = model.predict(img)[0][0]
        return "Pneumonia" if prediction >= 0.5 else "Normal"
    except Exception as e:
        return f"Model Error: {str(e)}"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file was uploaded
    if 'image' not in request.files:
        return render_template('predict.html', prediction="Error: No file uploaded")
    
    file = request.files['image']
    
    # Validate file
    if file.filename == '':
        return render_template('predict.html', prediction="Error: No file uploaded")
    if not allowed_file(file.filename):
        return render_template('predict.html', prediction="Error: Only PNG/JPG/JPEG allowed")
    
    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Get prediction
    result = predict_pneumonia(file_path)
    
    # Render result
    return render_template('predict.html', 
                         prediction=result, 
                         image_path=file_path if not result.startswith("Error") else None)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=3000)