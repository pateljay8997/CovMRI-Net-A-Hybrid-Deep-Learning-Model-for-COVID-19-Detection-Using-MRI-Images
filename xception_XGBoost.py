import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Define paths
dataset_dir = 'D:/Btech_project/Covid/COVID-19_Radiography_Dataset'
save_dir = 'D:/Btech_project/Covid/xception_XGBoost'
os.makedirs(save_dir, exist_ok=True)

# Load and preprocess data (same as before)
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

if len(all_images) == 0:
    raise ValueError("No images found in the dataset directory!")

print(f"Total images found: {len(all_images)}")
print(f"Class distribution: {pd.Series(all_labels).value_counts()}")

# Split dataset
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Create DataFrames
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
test_df = pd.DataFrame({'filename': test_images, 'class': test_labels})

# Define ImageDataGenerators
datagen = ImageDataGenerator(rescale=1./255)

# Set image size for Xception
img_size = (299, 299)

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

# Define and create the Xception model for feature extraction
def create_xception_feature_extractor(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

feature_extractor = create_xception_feature_extractor((img_size[0], img_size[1], 3))

# Extract features
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

print("Extracting features for training set...")
train_features, train_labels = extract_features(train_generator, feature_extractor)
print("Extracting features for test set...")
test_features, test_labels = extract_features(test_generator, feature_extractor)

# Convert labels to single column
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Train XGBoost
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(train_features, train_labels)

# Make predictions
y_pred = xgb_model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, y_pred, target_names=test_generator.class_indices.keys()))

# Save the models
feature_extractor.save(os.path.join(save_dir, 'xception_feature_extractor.keras'))
xgb_model.save_model(os.path.join(save_dir, 'xgboost_model.json'))

print(f"\nModels saved in: {save_dir}")
print("Feature extractor: xception_feature_extractor.keras")
print("XGBoost model: xgboost_model.json")