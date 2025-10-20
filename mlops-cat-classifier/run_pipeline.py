"""
Script to compile and submit the Cat Classifier training pipeline to Vertex AI
"""

import google.cloud.aiplatform as aip
from kfp import compiler
from pipeline import cat_classifier_pipeline_gcs, cat_classifier_pipeline_url

# ============================================================================
# Configuration - UPDATE THESE VALUES
# ============================================================================
PROJECT_ID = "de2025paritosh"
REGION = "us-central1"
PIPELINE_ROOT = "gs://temp_catclassifier"  # Your temp bucket for pipeline artifacts
MODEL_BUCKET = "model_catclassifier"  # Your model bucket (without gs://)
DATA_BUCKET = "data_catclassifier"  # Your data bucket (without gs://)
DATASET_ZIP_FILENAME = "cats_and_dogs_filtered.zip"  # Name of zip file in data bucket

# Pipeline parameters
MIN_ACCURACY_THRESHOLD = 0.75  # Minimum validation accuracy required
USE_GCS_DATA = True  # Set to True to use data from GCS bucket, False to use public URL


def compile_pipeline():
    """Compile the pipeline to YAML"""
    print("Compiling pipeline...")

    if USE_GCS_DATA:
        compiler.Compiler().compile(
            pipeline_func=cat_classifier_pipeline_gcs,
            package_path='cat_classifier_training_pipeline_gcs.yaml'
        )
        print("✓ GCS-based pipeline compiled to: cat_classifier_training_pipeline_gcs.yaml")
        return 'cat_classifier_training_pipeline_gcs.yaml'
    else:
        compiler.Compiler().compile(
            pipeline_func=cat_classifier_pipeline_url,
            package_path='cat_classifier_training_pipeline_url.yaml'
        )
        print("✓ URL-based pipeline compiled to: cat_classifier_training_pipeline_url.yaml")
        return 'cat_classifier_training_pipeline_url.yaml'


def submit_pipeline(template_path):
    """Submit the pipeline to Vertex AI for execution"""
    print(f"\nInitializing Vertex AI with project: {PROJECT_ID}")

    aip.init(
        project=PROJECT_ID,
        staging_bucket=PIPELINE_ROOT,
        location=REGION
    )

    print("Creating pipeline job...")

    if USE_GCS_DATA:
        parameter_values = {
            'project_id': PROJECT_ID,
            'data_bucket': DATA_BUCKET,
            'dataset_zip_filename': DATASET_ZIP_FILENAME,
            'model_bucket': MODEL_BUCKET,
            'min_accuracy_threshold': MIN_ACCURACY_THRESHOLD
        }
        print(f"Data source: gs://{DATA_BUCKET}/{DATASET_ZIP_FILENAME}")
    else:
        parameter_values = {
            'project_id': PROJECT_ID,
            'model_bucket': MODEL_BUCKET,
            'min_accuracy_threshold': MIN_ACCURACY_THRESHOLD
        }
        print("Data source: Public URL (https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)")

    job = aip.PipelineJob(
        display_name="cat-classifier-training-pipeline",
        enable_caching=False,  # Disable caching for training runs
        template_path=template_path,
        pipeline_root=PIPELINE_ROOT,
        location=REGION,
        parameter_values=parameter_values
    )

    print("Submitting pipeline to Vertex AI...")
    print(f"Pipeline will run in: {REGION}")
    print(f"Artifacts will be stored in: {PIPELINE_ROOT}")
    print(f"Model will be uploaded to: gs://{MODEL_BUCKET}")
    print("\n" + "="*60)

    job.run()

    print("\n" + "="*60)
    print("✓ Pipeline execution completed!")


if __name__ == "__main__":
    import sys

    # Compile the pipeline
    template_path = compile_pipeline()

    # Ask user if they want to submit
    response = input("\nDo you want to submit the pipeline to Vertex AI? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        submit_pipeline(template_path)
    else:
        print("\nPipeline compiled but not submitted.")
        print("To submit later, run this script again or use the Vertex AI console.")
