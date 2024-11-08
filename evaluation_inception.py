import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def find_model_file(directory='.'):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.keras'):
                return os.path.join(root, file)
    return None

def find_dataset_directory(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "COVID-19_Radiography_Dataset" in dirs:
            return os.path.join(root, "COVID-19_Radiography_Dataset")
    return None

# Create a directory to save evaluation results
results_dir = 'evaluation_results_inception'
os.makedirs(results_dir, exist_ok=True)

# Try to load the trained model
model_path = find_model_file()
if model_path is None:
    print("Error: No .keras model file found.")
    print("Please ensure that the model file is in the current directory or its subdirectories.")
    sys.exit(1)

try:
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit(1)

# Find the dataset directory
dataset_dir = find_dataset_directory('/')
if dataset_dir is None:
    print("Error: COVID-19_Radiography_Dataset not found.")
    print("Please provide the correct path to the dataset.")
    sys.exit(1)

print(f"Using dataset from: {dataset_dir}")

# Get all image files and their labels
all_images = []
all_labels = []

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        sub_dir = os.path.join(class_dir, "images")
        if os.path.isdir(sub_dir):
            for file_name in os.listdir(sub_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(sub_dir, file_name))
                    all_labels.append(class_name)

print(f"Total images found: {len(all_images)}")
print(f"Class distribution: {pd.Series(all_labels).value_counts()}")

# Create a DataFrame for the dataset
df = pd.DataFrame({'filename': all_images, 'class': all_labels})

# Create a data generator
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for maintaining order
)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(generator)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Get predictions
print("Generating predictions...")
y_pred = model.predict(generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = generator.classes

# Calculate metrics
print("Calculating metrics...")
f1 = f1_score(y_true, y_pred_classes, average='weighted')
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
cm = confusion_matrix(y_true, y_pred_classes)

# Calculate sensitivity and specificity for each class
sensitivities = []
specificities = []
for i in range(len(generator.class_indices)):
    tp = cm[i, i]
    fn = np.sum(cm[i, :]) - tp
    tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
    fp = np.sum(cm[:, i]) - tp
    sensitivities.append(tp / (tp + fn))
    specificities.append(tn / (tn + fp))

# Save results to a text file
print("Saving results...")
with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
    f.write(f"Model file: {model_path}\n")
    f.write(f"Dataset directory: {dataset_dir}\n\n")
    f.write(f"Loss: {loss:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write("Sensitivities per class:\n")
    for i, sens in enumerate(sensitivities):
        f.write(f"  Class {i}: {sens:.4f}\n")
    f.write("Specificities per class:\n")
    for i, spec in enumerate(specificities):
        f.write(f"  Class {i}: {spec:.4f}\n")

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))

# Save classification report
report = classification_report(y_true, y_pred_classes, target_names=generator.class_indices.keys())
with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

print("Evaluation completed. Results saved in 'evaluation_results' directory.")