from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# ---- Load own model ----
MODEL_PATH = os.path.join("model", "cats_vs_dogs_model.keras")

print(f" Loading model from {MODEL_PATH} ...")

try:
    model = load_model(MODEL_PATH)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    raise e

def prepare_image(img):
    """Preprocess the uploaded image for model prediction"""
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((224, 224))  # match training image size
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0  # normalize
    return arr

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "cats_vs_dogs_model"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))
    arr = prepare_image(img)

    preds = model.predict(arr)
    confidence = float(preds[0][0])  # assuming model outputs single sigmoid neuron

    is_cat = confidence < 0.5
    return jsonify({
        "is_cat": bool(is_cat),
        "confidence": confidence if is_cat else 1 - confidence,
        "detected_class": "cat" if is_cat else "not cat"
    })

if __name__ == "__main__":
    # Cloud Run sets $PORT automatically
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Starting Cat Predictor API on port {port}...")
    ## IMPORTANT: host must be 0.0.0.0 so Cloud Run can reach it
    app.run(host="0.0.0.0", port=port)
