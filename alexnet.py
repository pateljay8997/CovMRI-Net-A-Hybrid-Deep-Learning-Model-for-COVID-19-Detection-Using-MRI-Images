import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# Define the path to the dataset
dataset_dir = 'D:/Btech_project/Covid/COVID-19_Radiography_Dataset'

# Define the path to save the model
save_dir = '/alexnet'

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Get all image files and their labels
all_images = []
all_labels = []

# Collect all images and their labels
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        sub_dir = os.path.join(class_dir, "images")
        if os.path.isdir(sub_dir):
            for file_name in os.listdir(sub_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(sub_dir, file_name))
                    all_labels.append(class_name)

# Ensure images were found
if len(all_images) == 0:
    raise ValueError("No images found in the dataset directory!")

print(f"Total images found: {len(all_images)}")
print(f"Class distribution: {pd.Series(all_labels).value_counts()}")

# Split dataset into train and validation sets
train_images, validation_images, train_labels, validation_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Create DataFrames for train and validation sets
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
validation_df = pd.DataFrame({'filename': validation_images, 'class': validation_labels})

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(validation_df)}")

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Set image size for AlexNet (traditionally 227x227, but we can use 224x224)
img_size = (224, 224)

# Create data generators
batch_size = 32
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define AlexNet model
def create_alexnet(input_shape, num_classes):
    model = Sequential([
        Conv2D(96, 11, strides=4, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(3, strides=2),
        Conv2D(256, 5, padding='same', activation='relu'),
        MaxPooling2D(3, strides=2),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPooling2D(3, strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile the model
num_classes = len(train_generator.class_indices)
model = create_alexnet((img_size[0], img_size[1], 3), num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Set up callbacks
checkpoint_path = os.path.join(save_dir, 'alexnet_best_model.keras')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

# Train the model
epochs = 2
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, checkpoint]
)

# Save the final model
final_model_path = os.path.join(save_dir, 'alexnet_final_model.keras')
try:
    model.save(final_model_path)
    print(f"Final model saved successfully at: {final_model_path}")
except Exception as e:
    print(f"Error saving the final model: {str(e)}")

# Print training summary
print("Training History:")
for epoch, values in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'],
                                   history.history['loss'], history.history['val_loss'])):
    print(f"Epoch {epoch+1}:")
    print(f"  Train Accuracy: {values[0]:.4f}")
    print(f"  Validation Accuracy: {values[1]:.4f}")
    print(f"  Train Loss: {values[2]:.4f}")
    print(f"  Validation Loss: {values[3]:.4f}")

print(f"\nTraining completed. Models saved in: {save_dir}")
print(f"Best model: {checkpoint_path}")
print(f"Final model: {final_model_path}")