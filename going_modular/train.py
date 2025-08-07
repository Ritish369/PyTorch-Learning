"""
Trains a PyTorch image classification model using device-agnostic code and argument flags enabled.
"""

import os
import torch
import data_setup, model_builder, engine, utils
from torchvision import transforms

# For command line argument flags
import argparse

def main():
    parser = argparse.ArgumentParser(description = "Argument flags enabled")

    parser.add_argument('--train_dir', default = 'data/pizza_steak_sushi/train', type = str)
    parser.add_argument('--test_dir', default = 'data/pizza_steak_sushi/test', type=str)
    parser.add_argument('--learning_rate', default = 0.001, type=float)
    parser.add_argument('--batch_size', default = 32, type=int)
    parser.add_argument('--epochs', default = 5, type = int)
    parser.add_argument('--hidden_units', default = 10, type = int)

    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate
    print(f"[INFO] Training model for {NUM_EPOCHS} with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and an lr of {LEARNING_RATE}")

    # Setup train and test directories
    train_dir = args.train_dir
    test_dir = args.test_dir
    print(f"[INFO] Train dir is: {train_dir}")
    print(f"[INFO] Test dir is: {test_dir}")

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    # Create DataLoaders using data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform, BATCH_SIZE)

    # Create model using model_builder.py
    model = model_builder.TinyVGG(3, HIDDEN_UNITS, len(class_names)).to(device)

    # Setup loss fn and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training using engine.py
    engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, NUM_EPOCHS, device)

    # Save the model using utils.py
    utils.save_model(model, "models", "05_going_modular_script_mode_tinyvgg_model.pth")

# Required on windows to avoid endless running
if __name__ == "__main__":
    main()
