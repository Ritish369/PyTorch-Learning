# Step 1
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

## Setup class names
class_names = ["pizza", "steak", "sushi"]

# Step 2
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names))
effnetb2.load_state_dict(torch.load(f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
                                   map_location=torch.device("cpu"), weights_only = True))

# Step 3
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Timer start
    start_time = timer()

    # Transform the image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Get model into eval() mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass transformed image through the model and turn pred logits to pred probs
        pred_logits = effnetb2(img)
        pred_probs = torch.softmax(pred_logits, dim = 1)

    # Create pred label and pred prob dict for each pred class (this is the reqd format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the pred time
    pred_time = round(timer() - start_time, 5)

    # return pred dict and pred time
    return pred_labels_and_probs, pred_time   

# Step 4
## Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

## Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

## Create the Gradio demo
demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

## Launch the demo
demo.launch()
