from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import sys
import logging

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---- Load own model ----
MODEL_PATH = os.path.join("model", "model.keras")

logger.info(f"Loading model from {MODEL_PATH} ...")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Model directory exists: {os.path.exists('model')}")
if os.path.exists('model'):
    logger.info(f"Files in model directory: {os.listdir('model')}")

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}", exc_info=True)
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
    import time
    start_time = time.time()
    logger.info("Predict endpoint called")

    if "file" not in request.files:
        logger.error("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    logger.info(f"File received: {file.filename}")

    try:
        logger.info("Opening image...")
        img = Image.open(io.BytesIO(file.read()))
        logger.info(f"Image opened: size={img.size}, mode={img.mode}")

        logger.info("Preparing image...")
        arr = prepare_image(img)
        logger.info(f"Image prepared: shape={arr.shape}")

        logger.info("Running model prediction...")
        pred_start = time.time()
        preds = model.predict(arr)
        pred_time = time.time() - pred_start
        logger.info(f"Prediction completed in {pred_time:.2f}s")

        confidence = float(preds[0][0])
        is_cat = confidence < 0.5

        total_time = time.time() - start_time
        logger.info(f"Total request time: {total_time:.2f}s")

        return jsonify({
            "is_cat": bool(is_cat),
            "confidence": confidence if is_cat else 1 - confidence,
            "detected_class": "cat" if is_cat else "not cat"
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Cloud Run sets $PORT automatically
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Starting Cat Predictor API on port {port}...")
    ## IMPORTANT: host must be 0.0.0.0 so Cloud Run can reach it
    app.run(host="0.0.0.0", port=port)
