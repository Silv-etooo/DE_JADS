mlops-cat-classifier/ <br>
├── notebooks/ <br>
│ └── cat_classifier.ipynb # Jupyter notebook for developing/testing the model <br>
│ <br>
├──app/
    ├── api/
    │   ├── Dockerfile.api
    │   ├── predict_api.py
    │   ├── model/
    │   │   └── cats_vs_dogs_model.keras
    └── ui/
        ├── Dockerfile.ui
        └── ui_server.py
│ ├── pipeline.py #(->run to generate the vertex_training.yaml) # Vertex AI pipeline definition (KFP) <br>
│ ├── vertex_training.yaml # Vertex AI for continous training <br>
│ <br>
├── Dockerfile.ui # Dockerfile for the training component <br>
├── Dockerfile.api # Dockerfile for the prediction API service <br>
├── x (remove -> has to be split into two) cloudbuild.yaml # CI/CD config for GCP Cloud Build <br>
├── cicd_training.yaml # CI/CD training trat triggers vertex ai<br>
├── cicd_deployment.yaml # CI/CD deploys to api <br>
│ <br>
├── requirements.txt # Python dependencies <br>
├── .gitignore # Ignore unnecessary files (e.g., pycache, .DS_Store, etc.) <br>
└── README.md # You're here! Overview, setup, and usage instructions <br>
 