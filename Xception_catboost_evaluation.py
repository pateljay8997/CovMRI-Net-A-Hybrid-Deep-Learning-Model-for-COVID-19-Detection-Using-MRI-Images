import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from catboost import CatBoostClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths for model, data, and save directory
model_path = r"C:\Users\lumen\OneDrive\Desktop\jay.patel_dataset\Covid (1)\Covid\DATA\catboost_model.cbm"
history_path = os.path.join(r"C:\Users\lumen\OneDrive\Desktop\jay.patel_dataset\Covid (1)\Covid\DATA", 'training_history.pkl')
save_dir = r"C:\Users\lumen\OneDrive\Desktop\jay.patel_dataset\Covid (1)\Covid\DATA"

# Function to save plots
def save_plot(plt_obj, filename):
    """Save the plot to the specified directory."""
    path = os.path.join(save_dir, filename)
    plt_obj.savefig(path)
    plt_obj.close()
    print(f"Saved {filename}")

# Load the trained model
model = CatBoostClassifier()
model.load_model(model_path)

# Load your test data (modify as needed)
test_data_dir = r"C:\Users\lumen\OneDrive\Desktop\jay.patel_dataset\Covid (1)\Covid\COVID-19_Radiography_Dataset"  # Adjust this path
image_generator = ImageDataGenerator(rescale=1./255)

# Create a generator for test data
test_generator = image_generator.flow_from_directory(
    test_data_dir,
    target_size=(299, 299),  # Adjust according to your model input
    class_mode='binary',      # Change as needed
    batch_size=32,
    shuffle=False,
    seed=42  # Set a seed for reproducibility
)

# Get the true labels from the generator
y_test_labels = test_generator.classes

# Prepare to collect predictions
y_pred_proba = []

# Loop through the generator to make predictions in batches
for _ in range(len(test_generator)):
    # Get a batch of images
    batch_x, _ = next(test_generator)
    # Predict probabilities for this batch
    batch_pred = model.predict_proba(batch_x)
    y_pred_proba.append(batch_pred)

# Concatenate all batch predictions into a single array
y_pred_proba = np.vstack(y_pred_proba)

# Load Training History
history = None
try:
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    print("Training history loaded successfully.")
except FileNotFoundError:
    print("Training history file not found. Skipping some plots.")
except pickle.UnpicklingError as e:
    print(f"Error loading the pickle file: {e}. The file might be corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Generate Training & Validation Accuracy/Loss Graphs if history is available
if history:
    plt.figure()
    plt.plot(history['iterations'], history['learn']['Logloss'], label='Training Loss')
    if 'validation' in history:
        plt.plot(history['iterations'], history['validation']['Logloss'], label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Logloss')
    plt.title('Training and Validation Loss')
    plt.legend()
    save_plot(plt, 'training_validation_loss.png')

    plt.figure()
    plt.plot(history['iterations'], history['learn']['AUC'], label='Training Accuracy (AUC)')
    if 'validation' in history:
        plt.plot(history['iterations'], history['validation']['AUC'], label='Validation Accuracy (AUC)')
    plt.xlabel('Iterations')
    plt.ylabel('AUC')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    save_plot(plt, 'training_validation_accuracy.png')
else:
    print("Skipping training and validation plots due to missing history.")

# Lift Curve Calculation and Plot
if 'y_pred_proba' in locals() and 'y_test_labels' in locals():
    print("Calculating lift curve.")
    sorted_indices = np.argsort(y_pred_proba[:, 1])[::-1]
    sorted_labels = y_test_labels[sorted_indices]
    sorted_preds = y_pred_proba[:, 1][sorted_indices]

    cumulative_gains = np.cumsum(sorted_labels)
    percentage_total = cumulative_gains / cumulative_gains[-1]

    plt.figure()
    plt.plot(np.arange(len(sorted_labels)) / len(sorted_labels), percentage_total, label="Lift Curve")
    plt.plot([0, 1], [0, 1], '--', color='gray', label="Random Guess")
    plt.xlabel("Proportion of Samples")
    plt.ylabel("Proportion of Positives Captured")
    plt.title("Lift Curve")
    plt.legend()
    save_plot(plt, 'lift_curve.png')
else:
    print("Error: y_pred_proba or y_test_labels not defined. Cannot calculate lift curve.")

print("Evaluation completed. All results and graphs saved to the specified directory.")
