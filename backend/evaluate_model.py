import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("ml_model/drowsiness_model.h5")

# Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/test",
    image_size=(224, 224),   # Change according to your model input size
    batch_size=32,
    shuffle=False
)

# True labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Predictions
y_pred_prob = model.predict(test_dataset)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Awake", "Drowsy"]
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Awake", "Drowsy"],
    yticklabels=["Awake", "Drowsy"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()