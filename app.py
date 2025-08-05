from flask import Flask, request, render_template
from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import cv2

model = load_model('our_model.h5')

def predict_pneumonia(file):
    img = cv2.imread(file)

    if not is_xray(img):
        return "Not an X-ray"  
    
    img = cv2.resize(img,(300,300))
    img = img/255.0
    img = img.reshape(1,300,300,3)
    prediction = model.predict(img)
    return "Pneumonia" if prediction >= 0.5 else "Normal"

def is_xray(img):
    # Convert to grayscale if needed
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Check if image is mostly dark with bright regions (like an X-ray)
    mean_pixel = np.mean(gray)
    if mean_pixel < 50 or mean_pixel > 200:  # Adjust thresholds
        return False
    return True

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         file = request.files['image']
#         filename = file.filename
#         file_path = os.path.join('static/database', filename)
#         file.save(file_path)
#         result =predict_pneumonia(file_path)
#         if result:
#             prediction = "Pneumonia"
#         else:
#             prediction = "Normal"
#         return render_template('predict.html', prediction=prediction, image_path=file_path)
#     except:
#         prediction="File not found"
#         return render_template('predict.html', prediction=prediction)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/database', filename)
        file.save(file_path)
        
        result = predict_pneumonia(file_path)
        
        if result == "Not an X-ray":
            return render_template('predict.html', prediction="Error: Not a valid X-ray image", image_path=file_path)
        else:
            return render_template('predict.html', prediction=result, image_path=file_path)
    except:
        return render_template('predict.html', prediction="Error: File not found or invalid")
    
if __name__ == "__main__":
    app.run(debug=True, port= 5000)
    