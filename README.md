mlops-cat-classifier/
├── notebooks/
│   └── cat_classifier.ipynb
│       - Jupyter notebook for developing and testing the cat vs not-cat classifier.
│
├── src/
│   ├── train.py
│       - Python script to train the image classification model (based on MobileNetV2).
│   ├── pipeline.py
│       - Vertex AI pipeline definition using the KFP SDK.
│   ├── predict_api.py
│       - FastAPI app that serves the trained model as a REST API.
│
├── ui/
│   └── streamlit_app.py
│       - Frontend UI for uploading images and viewing predictions (Streamlit).
│
├── Dockerfile.train
│   - Dockerfile for building the training component.
│
├── Dockerfile.api
│   - Dockerfile for building the prediction API service.
│
├── cloudbuild.yaml
│   - Cloud Build CI/CD configuration to automate:
│       - building Docker images
│       - deploying training pipeline to Vertex AI
│       - deploying the API service
│
├── requirements.txt
│   - Python dependencies for training and API.
│
├── .gitignore
│   - Specifies files and folders to ignore in version control (e.g., __pycache__, model checkpoints).
│
├── README.md
│   - Project overview, setup instructions, usage, and links.
