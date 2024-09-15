import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractionModel(nn.Module):
    def __init__(self, num_classes):
        super(FeatureExtractionModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)  # Use ResNet for feature extraction
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
