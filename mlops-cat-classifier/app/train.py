import os
import zipfile
import numpy as np
import tensorflow as tf

# =======================
# CONFIG
# =======================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
MODEL_PATH = "model/cats_vs_dogs_model.keras"

# Set random seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)

# =======================
# DOWNLOAD & EXTRACT
# =======================
print("Downloading dataset...")
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=url, extract=False)

extract_dir = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))

train_dir = os.path.join(extract_dir, "train")
val_dir = os.path.join(extract_dir, "validation")

# =======================
# LOAD & NORMALIZE DATA
# =======================
print("Loading datasets...")
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

train_ds = train_ds_raw.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(AUTOTUNE)
val_ds = val_ds_raw.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(AUTOTUNE)

# =======================
# MODEL BUILDING
# =======================
print("Building model...")

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

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# =======================
# TRAINING PHASE 1
# =======================
print("Training base model...")
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# =======================
# FINE-TUNING
# =======================
print("Fine-tuning last 30 layers...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.fit(train_ds, epochs=1, validation_data=val_ds)

# =======================
# SAVE MODEL
# =======================
print(f"Saving model to {MODEL_PATH}...")
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

print("Done!")
