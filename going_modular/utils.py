# Common practice to save the helper functions in a utils.py file (utilities)
"""
Contains various utility funcs for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory with model_name.

    Args:
        model: trained and evaluated model to be saved
        target_dir: place to store the model (path)
        model_name: name of the saved model file with an extension .pth or .pt
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok = True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pth or .pt extension"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"Model is being saved at {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
