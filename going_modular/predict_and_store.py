
import pathlib
import torch
import torchvision
from typing import List, Dict
from timeit import default_timer as timer
from PIL import Image
from tqdm.auto import tqdm

# 1. Create a function with its required parameters to return a list of dicts with sample, truth label, pred, pred proband pred time
def pred_and_store(test_paths: List[pathlib.Path], model: torch.nn.Module, transforms: torchvision.transforms,
                   class_names: List[str], device: str= "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
    # 2. Create an empty list to store prediction dictionaries
    pred_results_list = []
    # 3. Loop through the target input (test images) paths
    for path in tqdm(test_paths):
        # 4. Create an empty dict for each iteration to store pred values per sample
        pred_dict = {}
        # 5. Get sample path and ground truth class name
        pred_dict["image_path"] = path
        gnd_truth_class = path.parent.stem
        pred_dict["gnd_truth_class"] = gnd_truth_class
        # 6. Start the pred timer using timeit library
        start_time = timer()
        # 7. Open image in the path using PIL
        image = Image.open(path)
        # 8. Tranform image to be compatible with the PyTorch model, add a batch dimension and send the image to the target device
        transformed_image = transforms(image).unsqueeze(0).to(device)
        # 9. Prepare model for inference by turning on eval() mode and sending it to the target device
        model.to(device)
        model.eval()
        # 10. Turn on inference mode, pass the target transformed img to model, calculate pred prob and target label
        with torch.inference_mode():
            pred_logits = model(transformed_image)
            pred_probs = torch.softmax(pred_logits, dim = 1)
            pred_label = torch.argmax(pred_probs, dim = 1)
            pred_class = class_names[pred_label.cpu()]
            # 11. Add pred prob and pred class to the pred dict created in step 4; Make sure they are on CPU
            pred_dict["pred_probs"] = round(pred_probs.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class
            # 12. End the pred timer started in step 6 and add time to pred dict in step 4
            end_time = timer()
            elapsed_time = round(end_time - start_time, 4)
            pred_dict["pred_time"] = elapsed_time
        # 13. Check if gnd truth in step 5 and pred_class match, then add the result to pred dict in step 4
        pred_dict["gndtrth_predclass_match"] = (gnd_truth_class == pred_class)
        # 14. Append updated pred dict to empty list of predictions created in step 2
        pred_results_list.append(pred_dict)
    # 15. Return the list of pred dicts
    return pred_results_list
