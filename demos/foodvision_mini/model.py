import torch
import torchvision
from torch import nn

# Functionalize the EffNetB2 feature extractor model creation
def create_effnetb2_model(num_classes: int=3, seed: int=42):
    """Creates an EfficientNetB2 feature extractor model and its transforms.
    Returns the model and transforms.
    """
    # 1, 2, 3 Steps here
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Step 4
    for param in model.parameters():
        param.requires_grad = False

    # Step 5
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transforms
