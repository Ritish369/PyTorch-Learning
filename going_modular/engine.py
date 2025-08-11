# Because this is the main area where model will be trained and all -- so, engine.
"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Tuple, List, Dict

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Target PyTorch model turned to train mode and then runs through all reqd training steps.

    Args: All are self explainatory

    Returns:
    A tuple of training loss and training accuracy (train loss, train accuracy) metrics.
    """
    # Put model in train mode
    model.train()

    # Setup train loss and acc values
    train_loss, train_acc = 0, 0

    # Loop through the data batches of dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Put data to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3. Zero grad the optimizer
        optimizer.zero_grad()
        # 4. Backpropagation
        loss.backward()
        # 5. step the optimizer
        optimizer.step()

        # Calculate and accumulate the train accuracies over each batch
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics for avg train loss and acc per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Test a PyTorch model for a single epoch.

    Turns the model to eval mode and then performs forward pass on test dataset.

    Args: Self explainatory

    Returns:
    A tuple of testing loss and accuracy (test loss, test accuracy) metrics.
    """
    # Model in eval mode
    model.eval()

    # Setup test loss and acc values
    test_loss, test_acc = 0, 0

    # Turn on imference context manager
    with torch.inference_mode():
        # Loop through dataloader's batches
        for batch, (X, y) in enumerate(dataloader):
            # Data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)
            # 2. Calculate and accumulate loss
            test_loss += loss_fn(test_pred_logits, y).item()

            # Calc and accumulate test acc
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim = 1), dim = 1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

    # Adjust test loss and acc on avg per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device) -> Dict[str, List]:
    """Trains and tests the PyTorch model.

    Passes the target model through train_step() and test_step() fns for a number of epochs, training and testing the model in the same epoch loop.

    Calculates, prints and stores eval metrics throughout.

    Args: Self explainatory

    Returns:
    A dictionary of training and testing loss as well as accuracy metrics. Each metric has values in a list for each epoch.
    """
    # Create empty dictionary for results
    results = {"train_loss" : [], "train_acc" : [], "test_loss" : [], "test_acc" : []}

    # Loop through the number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # Print the stats
        print(f"\nEpoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}")

        # Update the results dict
        results["train_loss"].append(train_loss.detach().cpu() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss.detach().cpu() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc)

    return results
