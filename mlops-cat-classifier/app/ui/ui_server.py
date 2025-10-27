from flask import Flask, send_from_directory, request, jsonify
import os
import requests
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='templates', static_url_path='')


PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")

@app.route("/")
def index():
    # Serve your HTML UI (index.html in /templates)
    return send_from_directory("templates", "index.html")

@app.route("/predict", methods=["POST"])
def proxy_predict():
    """Proxy image uploads to the prediction API."""
    logger.info(f"Received prediction request")
    logger.info(f"Proxying to: {PREDICT_API_URL}")

    if "file" not in request.files:
        logger.error("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    logger.info(f"File received: {file.filename}, mimetype: {file.mimetype}")
    files = {"file": (file.filename, file.stream, file.mimetype)}

    try:
        logger.info(f"Sending POST request to {PREDICT_API_URL}")
        response = requests.post(PREDICT_API_URL, files=files, timeout=120)
        logger.info(f"API response status: {response.status_code}")
        logger.info(f"API response body: {response.text[:200]}")
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        logger.error(f"Proxy error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Proxy error: {str(e)}"}), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Cloud Run sets $PORT automatically
    print(f"🚀 Starting UI server on port {port}")
    print(f"🔗 Using API endpoint: {PREDICT_API_URL}")
    app.run(host="0.0.0.0", port=port)
