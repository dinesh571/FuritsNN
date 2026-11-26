import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DATASET_PATH = r"furits\fruits-360_100x100\fruits-360\Training"

IMG_SIZE = (100, 100)
BATCH_SIZE = 64
EPOCHS = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\nTotal Classes: {num_classes}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

base_model = MobileNetV2(
    input_shape=(100, 100, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save model
model.save("fruit_classifier.keras")

# ---------------------------
# Create Graphs
# ---------------------------

import matplotlib.pyplot as plt

# Accuracy Graph
plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy',
    linewidth=3
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy',
    linewidth=3
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.savefig("accuracy_graph.png")
plt.close()

# Loss Graph
plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss',
    linewidth=3
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss',
    linewidth=3
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("loss_graph.png")
plt.close()

print("✅ accuracy_graph.png created")
print("✅ loss_graph.png created")

model.save("fruit_classifier.keras")

print("\n✅ Model saved as fruit_classifier.keras")