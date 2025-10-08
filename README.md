mlops-cat-classifier/ <br>
├── notebooks/ <br>
│ └── cat_classifier.ipynb # Jupyter notebook for developing/testing the model <br>
│ <br>
├── src/ <br>
│ ├── train.py # Training script using TensorFlow + MobileNetV2 <br>
│ ├── pipeline.py # Vertex AI pipeline definition (KFP) <br>
│ └── predict_api.py # FastAPI service for serving predictions <br>
│ <br>
├── ui/ <br>
│ └── streamlit_app.py # (Optional) UI for uploading images and testing predictions <br>
│ <br>
├── Dockerfile.train # Dockerfile for the training component <br>
├── Dockerfile.api # Dockerfile for the prediction API service <br>
├── cloudbuild.yaml # CI/CD config for GCP Cloud Build <br>
│ <br>
├── requirements.txt # Python dependencies <br>
├── .gitignore # Ignore unnecessary files (e.g., pycache, .DS_Store, etc.) <br>
└── README.md # You're here! Overview, setup, and usage instructions <br>
