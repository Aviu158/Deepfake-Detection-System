import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

class DeepfakeModel:
    def __init__(self):
        self.model = load_model('deepfake_model.keras')
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img)
        if prediction[0] > 0.5:
            return 'Real'
        else:
            return 'Fake'
