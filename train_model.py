import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_dir = r"C:\Users\Hetal\OneDrive\Desktop\BCD\FYP Data\train"
val_dir = r"C:\Users\Hetal\OneDrive\Desktop\BCD\FYP Data\val"
output_model_path = r"C:\Users\Hetal\OneDrive\Desktop\BCD\model\breast_cancer_inceptionv3.h5"

def train_model():
    # Check if dataset paths exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Training or validation directory does not exist. Check dataset placement.")
    
    print("Loading data...")

    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299, 299),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(299, 299),
        batch_size=32,
        class_mode='binary'
    )
    
    print("Data loaded successfully!")

    # Model Initialization
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the base model layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    print("Model initialized!")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model compiled! Starting training...")

    # Training
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

    # Save model
    print(f"Saving model to {output_model_path}...")
    model.save(output_model_path)
    print("Model training complete and saved!")

if __name__ == "__main__":
    train_model()
