mlops-cat-classifier/ 
├── notebooks/ 
│ └── cat_classifier.ipynb # Jupyter notebook for developing/testing the model 
│ 
├── src/ 
│ ├── train.py # Training script using TensorFlow + MobileNetV2 
│ ├── pipeline.py # Vertex AI pipeline definition (KFP) 
│ └── predict_api.py # FastAPI service for serving predictions 
│ 
├── ui/ 
│ └── streamlit_app.py # (Optional) UI for uploading images and testing predictions 
│ 
├── Dockerfile.train # Dockerfile for the training component 
├── Dockerfile.api # Dockerfile for the prediction API service 
├── cloudbuild.yaml # CI/CD config for GCP Cloud Build 
│ 
├── requirements.txt # Python dependencies 
├── .gitignore # Ignore unnecessary files (e.g., pycache, .DS_Store, etc.) 
└── README.md # You're here! Overview, setup, and usage instructions 
