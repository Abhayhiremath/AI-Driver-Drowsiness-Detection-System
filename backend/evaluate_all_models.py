import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Path
test_path = "datasets/dataset_new/test"

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Models to Evaluate
models = {
    "ResNet50": "ml_model/resnet50_model.h5",
    "MobileNetV2": "ml_model/mobilenetv2_model.h5"
}

for model_name, model_path in models.items():

    print("\n" + "="*60)
    print(f"Evaluating {model_name}")
    print("="*60)

    # Load Model
    model = tf.keras.models.load_model(model_path)

    # Evaluate
    loss, accuracy = model.evaluate(test_data, verbose=0)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Loss: {loss:.4f}")

    # Predictions
    y_pred_prob = model.predict(test_data)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    y_true = test_data.classes

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Awake", "Drowsy"]
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Awake", "Drowsy"],
        yticklabels=["Awake", "Drowsy"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

print("\nAll Models Evaluated Successfully!")