import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Function to create dummy data
def create_dummy_data(base_dir):
    categories = ['train', 'valid']
    labels = ['real', 'fake']
    os.makedirs(base_dir, exist_ok=True)
    
    for category in categories:
        for label in labels:
            path = os.path.join(base_dir, category, label)
            os.makedirs(path, exist_ok=True)
            for i in range(10):  # Create 10 dummy images per category
                img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(path, f'{label}_{i}.jpg'), img)

# Create dummy data
create_dummy_data("D:/Deepfake/rvf10k")

# Function to load data
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary'
    )
    valid_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary'
    )
    return train_generator, valid_generator

# Load data
data_dir = "D:/Deepfake/rvf10k"
train_generator, valid_generator = load_data(data_dir)
print(train_generator,valid_generator)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    epochs=15
)

# Evaluate the model
loss, accuracy = model.evaluate(valid_generator)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

model.save('deepfake_model.keras')

# Function to detect deepfake images

def detect_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return 'Fake' if prediction[0] > 0.5 else 'Real'

print(detect_image("C:/Users/AdityaYadav/OneDrive/Desktop/Aditya1.jpg",model))
