# mnist_cnn.py
# Task 2: CNN on MNIST - Train, evaluate, and visualize 5 sample predictions.
# Save as: mnist_cnn.py
# Run: python mnist_cnn.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Directories
# -------------------------
OUT_DIR = "mnist_results"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load and preprocess MNIST
# -------------------------
# MNIST images: 28x28 grayscale, labels 0-9
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to [0,1], convert to float32, and add channel axis
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = np.expand_dims(x_train, -1)  # shape -> (N, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)

print("Training samples:", x_train.shape, "Test samples:", x_test.shape)

# -------------------------
# Build CNN model (simple, effective)
# -------------------------
def build_model(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.25),

        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model()
model.summary()

# -------------------------
# Compile
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# Callbacks: EarlyStopping and ModelCheckpoint
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
best_model_path = os.path.join(OUT_DIR, f"mnist_cnn_best_{timestamp}.h5")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# -------------------------
# Train
# -------------------------
EPOCHS = 20
BATCH_SIZE = 128

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# Save final model as well
final_model_path = os.path.join(OUT_DIR, f"mnist_cnn_final_{timestamp}.h5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
print(f"Best model saved to: {best_model_path}")

# -------------------------
# Evaluate on test set
# -------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Check that accuracy requirement is met
if test_acc >= 0.95:
    print("Requirement met: test accuracy >= 95%")
else:
    print("Requirement NOT met: test accuracy < 95%")

# -------------------------
# Predict and visualize 5 random samples from test set
# -------------------------
num_samples = 5
rng = np.random.default_rng(SEED)  # reproducible random picks
indices = rng.choice(len(x_test), size=num_samples, replace=False)

preds = model.predict(x_test[indices])
pred_labels = preds.argmax(axis=1)

for i, idx in enumerate(indices):
    img = x_test[idx].squeeze()  # (28,28)
    true_label = int(y_test[idx])
    pred_label = int(pred_labels[i])
    prob = float(np.max(preds[i]))

    # Plot and save to file (no interactive show)
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    title = f"True: {true_label}  Pred: {pred_label}  Prob: {prob:.3f}"
    plt.title(title)
    out_path = os.path.join(OUT_DIR, f"prediction_sample_{i}_idx{idx}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction image: {out_path}  -> {title}")

# -------------------------
# Optional: Print classification report and confusion matrix (small)
# -------------------------
from sklearn.metrics import classification_report, confusion_matrix

y_pred_test = model.predict(x_test).argmax(axis=1)
print("\nClassification Report (test set):")
print(classification_report(y_test, y_pred_test, digits=4))

cm = confusion_matrix(y_test, y_pred_test)
cm_path = os.path.join(OUT_DIR, "confusion_matrix.npy")
np.save(cm_path, cm)
print(f"Confusion matrix saved to: {cm_path}")

print("\nAll done. Results are in the folder:", OUT_DIR)
