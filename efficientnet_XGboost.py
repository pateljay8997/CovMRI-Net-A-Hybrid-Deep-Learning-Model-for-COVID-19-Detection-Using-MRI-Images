import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define paths
dataset_dir = 'D:/Btech_project/Covid/COVID-19_Radiography_Dataset'
save_dir = 'D:/Btech_project/Covid/efficientNet_XGboost'
os.makedirs(save_dir, exist_ok=True)

# Collect all images and their labels
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

# Split dataset
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Create DataFrames
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
test_df = pd.DataFrame({'filename': test_images, 'class': test_labels})

# Define ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Set image size for EfficientNetB0
img_size = (224, 224)

# Create data generators
batch_size = 32
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the EfficientNet model as feature extractor
def create_efficientnet_feature_extractor(input_shape):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Create the feature extractor
feature_extractor = create_efficientnet_feature_extractor((img_size[0], img_size[1], 3))

# Extract features
def extract_features(generator):
    features = []
    labels = []
    for i in range(len(generator)):
        x, y = generator[i]
        feat = feature_extractor.predict(x)
        features.append(feat)
        labels.append(y)
    return np.vstack(features), np.vstack(labels)

print("Extracting features from training set...")
X_train, y_train = extract_features(train_generator)
print("Extracting features from test set...")
X_test, y_test = extract_features(test_generator)

# Convert labels to single column
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# Train XGBoost
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=train_generator.class_indices.keys())
print("Classification Report:")
print(report)

# Save the hybrid model
feature_extractor_path = os.path.join(save_dir, 'efficientnet_feature_extractor.keras')
xgboost_model_path = os.path.join(save_dir, 'xgboost_model.joblib')

feature_extractor.save(feature_extractor_path, save_format='keras')
joblib.dump(xgb_model, xgboost_model_path)

print(f"Hybrid model saved:")
print(f"EfficientNet feature extractor: {feature_extractor_path}")
print(f"XGBoost model: {xgboost_model_path}")

# Save results
results_path = os.path.join(save_dir, 'hybrid_model_results.txt')
with open(results_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Results saved to: {results_path}")