mlops-cat-classifier/
├── notebooks/
│   └── cat_classifier.ipynb          # (your working notebook)
├── src/
│   ├── train.py                      # (training logic)
│   ├── pipeline.py                   # (Vertex AI pipeline definition)
│   ├── predict_api.py                # (FastAPI or Flask app)
├── ui/
│   └── streamlit_app.py              # (optional UI for image upload)
├── Dockerfile.train
├── Dockerfile.api
├── cloudbuild.yaml                  # (CI/CD for build + deploy)
├── README.md
├── requirements.txt
└── .gitignore
