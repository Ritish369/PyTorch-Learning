# 1. Imports and class names setup
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Dict, Tuple

## Setup class names
with open("class_names.txt", "r") as f:
    class_names = [food.strip() for food in f.readlines()]

# 2. Model and transforms preparation
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names))

## Load the saved model weights
effnetb2.load_state_dict(torch.load(f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
                                    weights_only=True, map_location=torch.device("cpu")))

# 3. Predict function
def predict(img) -> Tuple[Dict, float]:
    start_time = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim = 1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    end_time = timer()
    pred_time = round(end_time - start_time, 5)

    return pred_labels_and_probs, pred_time

# 4. Gradio app
title = "FoodVision Big 🍔👁"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food into [101 different classes](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes = 5, label="Predictions"), gr.Number(label="Prediction time (s)")],
                   examples = example_list, title=title, description=description, article=article)

demo.launch()
