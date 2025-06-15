# -*- coding: utf-8 -*-
# OOP-Compliant CNN for Soil Classification

import tensorflow as tf
tf.config.run_functions_eagerly(True)
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SoilClassifier:
    def __init__(self, data_dir, model_path=None):
        self.data_dir = data_dir
        self.class_names = os.listdir(self.data_dir)
        self.model = None

        if model_path:
            self.load_model(model_path)

    def load_data(self):
        """Loads images and labels from the dataset directory."""
        images = []
        labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)

                # Skip non-image files
                if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    print(f"Skipping non-image file: {image_path}")
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                image = cv2.resize(image, (224, 224))
                images.append(image)
                labels.append(self.class_names.index(class_name))

        images = np.array(images) / 255.0  # Normalize
        labels = np.array(labels)

        return train_test_split(images, labels, test_size=0.2, random_state=42)

    def build_model(self):
        """Builds and compiles a CNN model."""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, X_train, y_train, epochs=30, batch_size=32):
        """Trains the CNN model with data augmentation."""
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        datagen.fit(X_train)
        tf.config.run_functions_eagerly(True)

        self.model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs)

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model on test data."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=2)
        print(f"Test Accuracy: {accuracy:.4f}")

    def save_model(self, filename="soil_model.h5"):
        """Saves the trained model as a .h5 file."""
        self.model.save(filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename="soil_model.h5"):
        """Loads a previously saved model."""
        self.model = load_model(filename)
        print(f"Model loaded from {filename}")
    
    def predict_soil_type(self, image_input):
        import numpy as np
        import cv2

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
        else:
            image = image_input

        image = cv2.resize(image, (224, 224))  # ✅ Match training size
        image = image / 255.0
        image = image.reshape(1, 224, 224, 3)

        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction)

        return self.class_names[predicted_class]  # 🔄 You were using `self.label_map` before



# Optional: Train or test only if running directly
if __name__ == "__main__":
    data_dir = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/extras/Soil_Types"
    model_path = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/extras/soil_model.h5"

    classifier = SoilClassifier(data_dir)

    # Train and evaluate the model
    X_train, X_test, y_train, y_test = classifier.load_data()
    classifier.build_model()
    classifier.train_model(X_train, y_train)
    classifier.evaluate_model(X_test, y_test)
    classifier.save_model(model_path)

    # Test prediction
    test_image = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/extras/Soil_Types/Red/IMG20250223172113.jpg"
    classifier.load_model(model_path)
    prediction = classifier.predict_soil_type(test_image)
    print(f"Predicted soil type: {prediction}")
