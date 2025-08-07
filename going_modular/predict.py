
import torch
import torchvision
import os
import argparse
from torchvision import transforms
from typing import Tuple
import model_builder

parser = argparse.ArgumentParser()

parser.add_argument('--model_file_path', default = 'models/05_going_modular_script_mode_tinyvgg_model.pth', type = str)
parser.add_argument('--image', type = str)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def prediction(model_file_path: str=args.model_file_path, image_path: str=args.image, device: torch.device=device) -> Tuple[str, str]:

    class_names = ['pizza', 'steak', 'sushi']

    loaded_model = model_builder.TinyVGG(3, 128, len(class_names)).to(device)
    loaded_model.load_state_dict(torch.load(model_file_path, weights_only = True))

    loaded_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.

    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])

    transformed_image = transform(loaded_image).unsqueeze(dim=0).to(device)

    loaded_model.eval()
    with torch.inference_mode():
        logit = loaded_model(transformed_image)
        prob = torch.softmax(logit, dim = 1)
        pred_label = class_names[torch.argmax(prob, dim = 1).cpu()]
    orig_label = os.path.basename(args.image).split(".")[0]

    return pred_label, orig_label

if __name__ == "__main__":
    pred, og = prediction()
    print(f"Predicted class is: {pred}")
    print(f"Original class is: {og}")
