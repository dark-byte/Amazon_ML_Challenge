import torch
from src.model import FeatureExtractionModel
from src.preprocess import preprocess_image

def predict(image_path, model):
    """Predict the entity value for the given image using the trained model."""
    img = preprocess_image(image_path)
    img = torch.tensor(img).unsqueeze(0)  # Convert to tensor and add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(img)
    return output
