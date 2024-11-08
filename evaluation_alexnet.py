import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths
model_path = '/alexnet/alexnet_final_model.keras'
dataset_dir = 'D:/Btech_project/Covid/COVID-19_Radiography_Dataset'
results_dir = '/alexnet/evaluation_results'

# Create the results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Load the trained model
model = load_model(model_path)

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

# Create a DataFrame for the dataset
df = pd.DataFrame({'filename': all_images, 'class': all_labels})

# Create a data generator
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),  # AlexNet input size
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
        f.write(f"  Class {list(generator.class_indices.keys())[i]}: {sens:.4f}\n")
    f.write("Specificities per class:\n")
    for i, spec in enumerate(specificities):
        f.write(f"  Class {list(generator.class_indices.keys())[i]}: {spec:.4f}\n")

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

print(f"Evaluation completed. Results saved in '{results_dir}'")