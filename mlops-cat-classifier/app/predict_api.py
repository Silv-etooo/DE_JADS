from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

# Initalize Flask App
app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Pre-Trained MobileNetV2 Model 
model = MobileNetV2(weights='imagenet')
#model = load_model('model/cats_vs_dogs_model') <-has to be changed to our model

def prepare_image(img):
    # Convert to RGB if needed (handles grayscale, RGBA, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # MobileNetV2 expects 224x224 images
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array

def is_cat(predictions):

    # Cat-related ImageNet classes
    cat_classes = ['tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 
                   'Egyptian_cat', 'cougar', 'lynx', 'leopard', 'tiger']
    
    # Check top 5 predictions
    for pred in predictions[0]:
        class_name = pred[1]
        confidence = float(pred[2])
        
        # Check if any cat-related class is in predictions
        if any(cat_class in class_name for cat_class in cat_classes):
            return True, confidence, class_name
    
    return False, 0.0, "not a cat"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'healthy',
        'model' : 'MobileNetV2'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """"
    Prediction endpoint that accepts an image and returns whether it's a cat.
    Expects: Image file in the request
    Returns: JSON with prediction results
    """

    try:
        if 'file' not in request.files:
            return jsonify({
                'error' : 'No File Provided',
                'message' : 'Please upload an image file using the File Field'
            }), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'error' : 'No file selected',
                'message' : 'Please select a file to upload'
            }), 400
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        processed_img = prepare_image(img)

        predictions = model.predict(processed_img)

        decoded_predictions = decode_predictions(predictions, top=5)

        cat_detected, confidence, label = is_cat(decoded_predictions)

        top_predictions = [
            {
                'class' : pred[1],
                'confidence' : float(pred[2])
            }
            for pred in decode_predictions[0]
        ]

        return jsonify({
            'is_cat' : cat_detected,
            'confidence' : confidence,
            'detected_class' : label,
            'top_predictions' : top_predictions
        })
    
    except Exception as e:
        return jsonify({
            'error' : 'Prediction failed',
            'message' : str(e)
        }), 500
    
@app.route('/')
def home():
    """Serve the HTML UI"""
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask App
    app.run(host='0.0.0.0', port=8000, debug=True)