import cv2
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """Resize and normalize images for model input."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to 224x224 (for ResNet input size)
    img = np.array(img) / 255.0   # Normalize pixel values
    return img
