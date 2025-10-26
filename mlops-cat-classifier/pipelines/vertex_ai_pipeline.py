"""
Cat Classifier Training Pipeline using Kubeflow Pipelines SDK v2
This pipeline downloads data, trains a MobileNetV2 model, evaluates it, and uploads to GCS.
"""

import kfp
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
)
from typing import NamedTuple


# ============================================================================
# Pipeline Component 1: Data Ingestion from GCS
# ============================================================================
@dsl.component(
    packages_to_install=["google-cloud-storage", "tensorflow==2.15.0"],
    base_image="python:3.10-slim"
)
def download_cat_dog_data_from_gcs(
    project_id: str,
    data_bucket: str,
    dataset_zip_filename: str,
    dataset_output: Output[Dataset]
):
    """Download and prepare the cats vs dogs dataset from GCS bucket"""
    import os
    import zipfile
    from google.cloud import storage
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Download from GCS
    logging.info(f"Downloading dataset from gs://{data_bucket}/{dataset_zip_filename}...")
    client = storage.Client(project=project_id)
    bucket = client.bucket(data_bucket)
    blob = bucket.blob(dataset_zip_filename)

    # Create temp directory
    temp_dir = "/tmp/dataset"
    os.makedirs(temp_dir, exist_ok=True)

    zip_path = os.path.join(temp_dir, "cats_and_dogs_filtered.zip")
    blob.download_to_filename(zip_path)

    logging.info(f"Downloaded to {zip_path}")

    # Extract dataset directly to the KFP artifact path
    logging.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_output.path)

    # The zip contains cats_and_dogs_filtered folder at the root
    extract_dir = os.path.join(dataset_output.path, "cats_and_dogs_filtered")

    # Verify the train directory exists
    if not os.path.exists(os.path.join(extract_dir, "train")):
        raise FileNotFoundError(f"Train directory not found at {extract_dir}/train")

    logging.info(f"Dataset extracted to KFP artifact path: {dataset_output.path}")
    logging.info(f"Dataset directory: {extract_dir}")


# ============================================================================
# Pipeline Component 1b: Data Ingestion from Public URL (Fallback)
# ============================================================================
@dsl.component(
    packages_to_install=["tensorflow==2.15.0"],
    base_image="python:3.10-slim"
)
def download_cat_dog_data_from_url(
    dataset_output: Output[Dataset]
):
    """Download and prepare the cats vs dogs dataset from public URL"""
    import os
    import zipfile
    import tensorflow as tf
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logging.info("Downloading cats and dogs dataset from public URL...")
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_path = tf.keras.utils.get_file(
        "cats_and_dogs_filtered.zip",
        origin=url,
        extract=False
    )

    # Extract dataset directly to the KFP artifact path
    logging.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_output.path)

    # The zip contains cats_and_dogs_filtered folder at the root
    extract_dir = os.path.join(dataset_output.path, "cats_and_dogs_filtered")

    # Verify the train directory exists
    if not os.path.exists(os.path.join(extract_dir, "train")):
        raise FileNotFoundError(f"Train directory not found at {extract_dir}/train")

    logging.info(f"Dataset extracted to KFP artifact path: {dataset_output.path}")
    logging.info(f"Dataset directory: {extract_dir}")


# ============================================================================
# Pipeline Component 2: Model Training
# ============================================================================
@dsl.component(
    packages_to_install=["tensorflow==2.15.0", "numpy==1.24.3"],
    base_image="python:3.10-slim"
)
def train_cat_classifier(
    dataset_input: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics]
) -> NamedTuple('TrainingOutputs', train_accuracy=float, val_accuracy=float, val_loss=float):
    """Train MobileNetV2 transfer learning model for cat classification"""
    import os
    import tensorflow as tf
    import numpy as np
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 3
    SEED = 42
    AUTOTUNE = tf.data.AUTOTUNE

    # Set random seeds
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Read dataset from KFP artifact path
    # The dataset is extracted to dataset_input.path/cats_and_dogs_filtered
    extract_dir = os.path.join(dataset_input.path, "cats_and_dogs_filtered")

    train_dir = os.path.join(extract_dir, "train")
    val_dir = os.path.join(extract_dir, "validation")

    logging.info(f"Loading datasets from {extract_dir}...")

    # Load datasets
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        seed=SEED
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        seed=SEED
    )

    # Normalize datasets
    train_ds = train_ds_raw.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(AUTOTUNE)
    val_ds = val_ds_raw.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(AUTOTUNE)

    logging.info("Building MobileNetV2 transfer learning model...")

    # Build model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    logging.info("Training base model...")
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

    # Fine-tuning
    logging.info("Fine-tuning last 30 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    history_fine = model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=1)

    # Get final metrics
    final_train_acc = float(history_fine.history['accuracy'][-1])
    final_val_acc = float(history_fine.history['val_accuracy'][-1])
    final_val_loss = float(history_fine.history['val_loss'][-1])

    logging.info(f"Training Accuracy: {final_train_acc:.4f}")
    logging.info(f"Validation Accuracy: {final_val_acc:.4f}")
    logging.info(f"Validation Loss: {final_val_loss:.4f}")

    # Save model to KFP artifact path
    import os
    os.makedirs(model_output.path, exist_ok=True)

    model_output.metadata["framework"] = "tensorflow"
    model_output.metadata["model_type"] = "MobileNetV2"
    model_output.metadata["file_name"] = "model.keras"

    model_file = os.path.join(model_output.path, "model.keras")
    model.save(model_file)
    logging.info(f"Model saved to {model_file}")

    # Log metrics
    metrics_output.log_metric("train_accuracy", final_train_acc)
    metrics_output.log_metric("val_accuracy", final_val_acc)
    metrics_output.log_metric("val_loss", final_val_loss)

    outputs = NamedTuple(
        'TrainingOutputs',
        train_accuracy=float,
        val_accuracy=float,
        val_loss=float
    )

    return outputs(final_train_acc, final_val_acc, final_val_loss)


# ============================================================================
# Pipeline Component 3: Model Evaluation
# ============================================================================
@dsl.component(
    base_image="python:3.10-slim"
)
def evaluate_model(
    train_accuracy: float,
    val_accuracy: float,
    val_loss: float,
    min_accuracy_threshold: float = 0.75
) -> str:
    """Evaluate if model meets quality threshold"""
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logging.info(f"Model Evaluation:")
    logging.info(f"  Training Accuracy: {train_accuracy:.4f}")
    logging.info(f"  Validation Accuracy: {val_accuracy:.4f}")
    logging.info(f"  Validation Loss: {val_loss:.4f}")
    logging.info(f"  Minimum Threshold: {min_accuracy_threshold:.4f}")

    if val_accuracy >= min_accuracy_threshold:
        logging.info("✓ Model PASSED quality check!")
        return "PASS"
    else:
        logging.warning("✗ Model FAILED quality check!")
        return "FAIL"


# ============================================================================
# Pipeline Component 4: Upload Model to GCS
# ============================================================================
@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image="python:3.10-slim"
)
def upload_model_to_gcs(
    project_id: str,
    model_bucket: str,
    model: Input[Model],
    version: str = "v2"  # Force component refresh
):
    """Upload trained model to Google Cloud Storage - Version 2 with detailed logging"""
    from google.cloud import storage
    from google.cloud.exceptions import GoogleCloudError
    import logging
    import sys
    import os

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    try:
        # Debug: Print all inputs
        logging.info(f"=== Upload Model to GCS - Starting ===")
        logging.info(f"Project ID: {project_id}")
        logging.info(f"Model bucket: {model_bucket}")
        logging.info(f"Model artifact path: {model.path}")
        logging.info(f"Model metadata: {model.metadata}")

        # List contents of model artifact path
        logging.info(f"Listing contents of {model.path}:")
        if os.path.exists(model.path):
            for item in os.listdir(model.path):
                item_path = os.path.join(model.path, item)
                size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                logging.info(f"  - {item} ({'file' if os.path.isfile(item_path) else 'dir'}, {size} bytes)")
        else:
            logging.error(f"Model path does not exist: {model.path}")
            raise FileNotFoundError(f"Model artifact path not found: {model.path}")

        # Construct source file path from KFP artifact
        source_file = os.path.join(model.path, model.metadata.get('file_name', 'model.keras'))
        logging.info(f"Source file path: {source_file}")

        # Verify file exists
        if not os.path.exists(source_file):
            logging.error(f"Model file not found at {source_file}")
            raise FileNotFoundError(f"Model file not found at {source_file}")

        file_size = os.path.getsize(source_file)
        logging.info(f"Model file size: {file_size / (1024*1024):.2f} MB")

        # Initialize GCS client
        logging.info("Initializing GCS client...")
        client = storage.Client(project=project_id)

        # Check if bucket exists
        logging.info(f"Checking if bucket exists: {model_bucket}")
        bucket = client.bucket(model_bucket)
        if not bucket.exists():
            logging.error(f"Bucket does not exist: {model_bucket}")
            raise ValueError(f"Bucket does not exist: {model_bucket}")

        logging.info(f"Bucket {model_bucket} exists and is accessible")

        # Upload to GCS
        model_filename = "cat_classifier_model.keras"
        blob = bucket.blob(model_filename)

        logging.info(f"Starting upload to gs://{model_bucket}/{model_filename}...")
        blob.upload_from_filename(source_file)

        logging.info(f"✓ Model uploaded successfully to gs://{model_bucket}/{model_filename}")
        logging.info(f"✓ File size: {file_size / (1024*1024):.2f} MB")

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
        raise
    except GoogleCloudError as e:
        logging.error(f"Google Cloud error: {e}")
        logging.error(f"Error details: {type(e).__name__}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during upload: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


# ============================================================================
# Pipeline Component 5: Publish Pub/Sub Message
# ============================================================================
@dsl.component(
    packages_to_install=["google-cloud-pubsub"],
    base_image="python:3.10-slim"
)
def publish_model_trained_message(
    project_id: str,
    model_bucket: str,
    topic_id: str = "model-trained"
):
    """Publishes a Pub/Sub message announcing that model training is complete."""
    from google.cloud import pubsub_v1
    import json, datetime, logging, sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    message = {
        "model_bucket": model_bucket,
        "model_path": f"gs://{model_bucket}/cat_classifier_model.keras",
        "status": "TRAINING_COMPLETE",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

    message_bytes = json.dumps(message).encode("utf-8")
    future = publisher.publish(topic_path, message_bytes)
    future.result()  # wait for confirmation
    logging.info(f"✅ Published message to {topic_path}: {message}")



# ============================================================================
# Define the Pipeline (Using GCS Data Bucket)
# ============================================================================
@kfp.dsl.pipeline(
    name="cat-classifier-training-pipeline-gcs",
    description="Train and deploy a cat classifier using MobileNetV2 (data from GCS)"
)
def cat_classifier_pipeline_gcs(
    project_id: str,
    data_bucket: str,
    dataset_zip_filename: str,
    model_bucket: str,
    min_accuracy_threshold: float = 0.75
):
    """
    Cat Classifier Training Pipeline (GCS Data Source)

    Args:
        project_id: Google Cloud project ID
        data_bucket: GCS bucket containing the dataset zip file
        dataset_zip_filename: Name of the zip file in data_bucket
        model_bucket: GCS bucket name for storing trained models
        min_accuracy_threshold: Minimum validation accuracy required (default: 0.75)
    """

    # Step 1: Download data from GCS
    download_task = download_cat_dog_data_from_gcs(
        project_id=project_id,
        data_bucket=data_bucket,
        dataset_zip_filename=dataset_zip_filename
    )

    # Step 2: Train model
    train_task = train_cat_classifier(
        dataset_input=download_task.outputs["dataset_output"]
    )

    # Step 3: Evaluate model
    eval_task = evaluate_model(
        train_accuracy=train_task.outputs["train_accuracy"],
        val_accuracy=train_task.outputs["val_accuracy"],
        val_loss=train_task.outputs["val_loss"],
        min_accuracy_threshold=min_accuracy_threshold
    )

    # Step 4: Upload model to GCS (only if evaluation passes)
    with dsl.If(eval_task.output == "PASS"):
        upload_task = upload_model_to_gcs(
            project_id=project_id,
            model_bucket=model_bucket,
            model=train_task.outputs["model_output"],
            version="v2"
        )

        # Step 5: Publish Pub/Sub message to trigger deployment
        publish_message_task = publish_model_trained_message(
            project_id=project_id,
            model_bucket=model_bucket,
            topic_id="model-trained"   # must match your Cloud Build trigger topic
        ).after(upload_task)


# ============================================================================
# Define the Pipeline (Using Public URL) - Fallback
# ============================================================================
@kfp.dsl.pipeline(
    name="cat-classifier-training-pipeline-url",
    description="Train and deploy a cat classifier using MobileNetV2 (data from public URL)"
)
def cat_classifier_pipeline_url(
    project_id: str,
    model_bucket: str,
    min_accuracy_threshold: float = 0.75
):
    """
    Cat Classifier Training Pipeline (Public URL Data Source)

    Args:
        project_id: Google Cloud project ID
        model_bucket: GCS bucket name for storing trained models
        min_accuracy_threshold: Minimum validation accuracy required (default: 0.75)
    """

    # Step 1: Download data from public URL
    download_task = download_cat_dog_data_from_url()

    # Step 2: Train model
    train_task = train_cat_classifier(
        dataset_input=download_task.outputs["dataset_output"]
    )

    # Step 3: Evaluate model
    eval_task = evaluate_model(
        train_accuracy=train_task.outputs["train_accuracy"],
        val_accuracy=train_task.outputs["val_accuracy"],
        val_loss=train_task.outputs["val_loss"],
        min_accuracy_threshold=min_accuracy_threshold
    )

    # Step 4: Upload model to GCS (only if evaluation passes)
    with dsl.If(eval_task.output == "PASS"):
        upload_task = upload_model_to_gcs(
            project_id=project_id,
            model_bucket=model_bucket,
            model=train_task.outputs["model_output"],
            version="v2"
        )


# ============================================================================
# Compile the Pipeline
# ============================================================================
if __name__ == "__main__":
    from kfp import compiler
    import sys

    # Default: compile URL-based pipeline (no data upload required)
    pipeline_type = sys.argv[1] if len(sys.argv) > 1 else "url"

    if pipeline_type == "gcs":
        compiler.Compiler().compile(
            pipeline_func=cat_classifier_pipeline_gcs,
            package_path="mlops-cat-classifier/pipelines/cat_classifier_training_pipeline.yaml",
        )
        print("✓ GCS-based pipeline compiled to: mlops-cat-classifier/pipelines/cat_classifier_training_pipeline.yaml")
    else:
        compiler.Compiler().compile(
            pipeline_func=cat_classifier_pipeline_url,
            package_path="mlops-cat-classifier/pipelines/cat_classifier_training_pipeline.yaml",
        )
        print("✓ URL-based pipeline compiled to: mlops-cat-classifier/pipelines/cat_classifier_training_pipeline.yaml")
