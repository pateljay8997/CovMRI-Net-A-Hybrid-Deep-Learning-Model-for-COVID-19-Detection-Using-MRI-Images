# Import Libraries
import os
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

# Paths for Dataset and Save Directory
data_dir = r"C:\Users\lumen\OneDrive\Desktop\jay.patel_dataset\Covid (1)\Covid\COVID-19_Radiography_Dataset"
save_dir = r"C:\Users\lumen\OneDrive\Desktop\jay.patel_dataset\Covid (1)\Covid\DATA"

# Data Loading and Preprocessing
print("Step 1: Loading data and preprocessing images...")
datagen = ImageDataGenerator(rescale=1.0/255)
data_flow = datagen.flow_from_directory(
    data_dir, target_size=(299, 299), batch_size=32, class_mode='binary'
)
print("Step 1 completed.")

# Feature Extraction with Xception Model
print("Step 2: Extracting features using Xception model...")
xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

features, labels = [], []
for x_batch, y_batch in data_flow:
    features.append(xception_model.predict(x_batch))
    labels.extend(y_batch)
    if len(features) * 32 >= data_flow.samples:  # Limit to full dataset size
        break
features = np.concatenate(features)
labels = np.array(labels)
print("Step 2 completed.")

# Splitting Data for Training and Validation
print("Step 3: Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
print("Step 3 completed.")

# Model Training with CatBoost Classifier
print("Step 4: Initializing and training the CatBoost model...")
catboost_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.1, depth=6, eval_metric='AUC', verbose=100
)
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
print("Step 4 completed.")

# Save the Model and Training History
print("Step 5: Saving the trained model and training history...")
model_path = os.path.join(save_dir, 'catboost_model.cbm')
catboost_model.save_model(model_path)

history_path = os.path.join(save_dir, 'training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(catboost_model.get_evals_result(), f)
print("Step 5 completed.")

# Model Evaluation and Visualization
print("Step 6: Evaluating the model and generating plots...")

# ROC Curve
y_val_pred = catboost_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_val_pred)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
roc_path = os.path.join(save_dir, 'roc_curve.png')
plt.savefig(roc_path)
plt.show()
print("ROC Curve saved.")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_val_pred)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
prc_path = os.path.join(save_dir, 'precision_recall_curve.png')
plt.savefig(prc_path)
plt.show()
print("Precision-Recall Curve saved.")

# Lift Curve
sorted_indices = np.argsort(y_val_pred)[::-1]
sorted_labels = y_val[sorted_indices]
cumulative_gains = np.cumsum(sorted_labels) / np.sum(sorted_labels)
x_vals = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels)

plt.figure()
plt.plot(x_vals, cumulative_gains, label='Lift Curve')
plt.xlabel('Percentage of Samples')
plt.ylabel('Percentage of Positives')
plt.title('Lift Curve')
plt.legend()
lift_path = os.path.join(save_dir, 'lift_curve.png')
plt.savefig(lift_path)
plt.show()
print("Lift Curve saved.")

# Model Performance Metrics
print("Step 7: Displaying classification metrics and AUC score...")
print(classification_report(y_val, y_val_pred > 0.5))
print(f"AUC: {roc_auc_score(y_val, y_val_pred)}")

# Saving Validation Predictions and Labels
print("Step 8: Saving validation predictions and labels...")
preds_path = os.path.join(save_dir, 'y_val_pred.pkl')
with open(preds_path, 'wb') as f:
    pickle.dump((y_val, y_val_pred), f)
print("Step 8 completed. All steps finished.")
