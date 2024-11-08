import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
model_dir = 'D:/Btech_project/Covid/xception_xgboost'
save_dir = 'D:/Btech_project/Covid/xception_XGBoost'
dataset_dir = 'D:/Btech_project/Covid/COVID-19_Radiography_Dataset'

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load the Xception feature extractor
feature_extractor = load_model(os.path.join(model_dir, 'xception_feature_extractor.keras'))

# Load the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(model_dir, 'xgboost_model.json'))

# Prepare the dataset
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

# Split dataset into train and test sets
_, test_images, _, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Create DataFrame for test set
test_df = pd.DataFrame({'filename': test_images, 'class': test_labels})

# Set up the data generator
img_size = (299, 299)
datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Extract features from test data
def extract_features(generator, feature_extractor):
    features = []
    labels = []
    for i in range(len(generator)):
        x, y = generator[i]
        batch_features = feature_extractor.predict(x)
        features.append(batch_features)
        labels.append(y)
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(generator)} batches")
    return np.vstack(features), np.vstack(labels)

print("Extracting features for test set...")
test_features, test_labels = extract_features(test_generator, feature_extractor)

# Convert labels to single column
test_labels = np.argmax(test_labels, axis=1)

# Make predictions
y_pred = xgb_model.predict(test_features)

# Calculate metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='weighted')

# Generate classification report and confusion matrix
class_report = classification_report(test_labels, y_pred, target_names=test_generator.class_indices.keys())
conf_matrix = confusion_matrix(test_labels, y_pred)

# Save results
results = f"""
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}

Classification Report:
{class_report}

Confusion Matrix:
{conf_matrix}
"""

with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
    f.write(results)

print(f"Evaluation results saved to {os.path.join(save_dir, 'evaluation_results.txt')}")

# Save the results as a CSV for easier analysis
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision'],
    'Value': [accuracy, precision]
})
results_df.to_csv(os.path.join(save_dir, 'evaluation_metrics.csv'), index=False)

print(f"Evaluation metrics saved to {os.path.join(save_dir, 'evaluation_metrics.csv')}")

# Print results to console as well
print(results)