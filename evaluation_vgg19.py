import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model_path = '/vgg19_covid_model.h5'
model = load_model(model_path)

# Define paths and parameters
preprocessed_directory = 'D:\\Btech_project\\Covid\\images'
image_size = (224, 224)
batch_size = 32

# Create the data generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    directory=preprocessed_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,  # Shuffle should be False for evaluation
    subset='validation'
)

# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Predict the classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the ground truth classes
y_true = val_generator.classes

# Calculate metrics
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
accuracy = accuracy_score(y_true, y_pred_classes)

# Sensitivity (Recall) and Specificity
cm = confusion_matrix(y_true, y_pred_classes)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(f"Precision: {precision}")
print(f"Recall (Sensitivity): {recall}")
print(f"F1 Score: {f1}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys()))

# Confusion matrix
print("Confusion Matrix:")
print(cm)

# Display the confusion matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Save evaluation results to a file
results_file_path = '/evaluation_results.txt'
with open(results_file_path, 'w') as f:
    f.write(f"Validation Loss: {loss}\n")
    f.write(f"Validation Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall (Sensitivity): {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Sensitivity: {sensitivity}\n")
    f.write(f"Specificity: {specificity}\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys()))
    f.write("\nConfusion Matrix:\n")
    np.savetxt(f, cm, fmt='%d')

print(f"Evaluation completed and results saved to {results_file_path}.")
