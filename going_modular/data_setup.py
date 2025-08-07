# For creating script of the useful cell from the notebook; need to at the top level
"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
# Imports at the top of script being a separate program in itself
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int,
                      num_workers: int=NUM_WORKERS):
    """Creates training and testing DataLoaders.

    Takes in a train dir and test dir path and turns them into PyTorch Datasets ans then into PyTorch DataLoaders

    Args:
    train_dir, test_dir: self explainatory
    transform: torchvision transforms to perform on train and test data
    batch_size, num_workers: self explainatory

    Returns:
    Tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images (datasets) into Data Loaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle = True, num_workers=num_workers,
                                 pin_memory=True) # pin_memory copies the data tensors to the device before returning the output
    test_dataloader = DataLoader(test_data, batch_size, shuffle = False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names
