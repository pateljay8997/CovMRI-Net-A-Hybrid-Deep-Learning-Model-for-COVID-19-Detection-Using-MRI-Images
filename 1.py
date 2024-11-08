import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import mixed_precision
import pandas as pd
import numpy as np

# Enable mixed precision training for faster computation
mixed_precision.set_global_policy('mixed_float16')

# Define the path to the dataset
dataset_dir = 'D:/Btech_project/Covid/COVID-19_Radiography_Dataset'

# Get all image files and their labels
all_images = []
all_labels = []

# Map class directories to labels
class_names = sorted(os.listdir(dataset_dir))
class_map = {class_name: index for index, class_name in enumerate(class_names)}

# Collect all images and their labels
for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # Look for images in the subdirectory named "images"
        sub_dir = os.path.join(class_dir, "images")
        if os.path.isdir(sub_dir):
            for file_name in os.listdir(sub_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(sub_dir, file_name))
                    all_labels.append(class_name)  # Use class name instead of integer

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
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Create data generators
batch_size = 16  # Reduced batch size
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# Use len(train_generator.class_indices) to get the number of classes
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('inceptionv3_best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Train model
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=2,  # Reduced to 2 epochs
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping, checkpoint, tensorboard],
        verbose=1
    )

    # Save the final model
    model.save('inceptionv3_final_model.keras')

    # Print training summary
    print("Training History:")
    for epoch, values in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'],
                                       history.history['loss'], history.history['val_loss'])):
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Accuracy: {values[0]:.4f}")
        print(f"  Validation Accuracy: {values[1]:.4f}")
        print(f"  Train Loss: {values[2]:.4f}")
        print(f"  Validation Loss: {values[3]:.4f}")

except Exception as e:
    print(f"An error occurred during training: {str(e)}")

    # Try to get more information about the error
    import traceback

    traceback.print_exc()

    # Check if we can access the last batch of data
    try:
        last_batch = next(train_generator)
        print(f"Last batch shape: {last_batch[0].shape}")
        print(f"Last batch label shape: {last_batch[1].shape}")
        print(f"Last batch label values: {np.unique(np.argmax(last_batch[1], axis=1))}")
    except Exception as batch_error:
        print(f"Error accessing last batch: {str(batch_error)}")

print("Script completed.")