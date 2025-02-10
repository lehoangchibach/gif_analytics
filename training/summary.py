import argparse

import torch
from torchsummary import summary

from constants import *
from model import VideoCNN


def print_model_summary(model_path):
    """
    Load a saved PyTorch model and print its summary.

    Args:
        model_path (str): Path to the saved model checkpoint
        num_classes (int): Number of classes in the model
    """
    # Check if CUDA is available
    device = torch.device("cpu")

    # Recreate the model architecture
    checkpoint = torch.load(model_path, map_location="cpu")
    num_classes = checkpoint["num_classes"]
    model = VideoCNN(num_classes=num_classes).to(device)

    # Print model summary
    print(f"\nModel Summary")
    print("-" * 50)

    # Use torchsummary to print detailed model summary
    # Assumes input shape matches original training configuration
    summary(model, (NUM_FRAMES, *IMAGE_SIZE, 3), batch_size=-1, device="cpu")

    # Additional model information
    print("\nAdditional Model Information:")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Print PyTorch Model Summary")
    parser.add_argument("model_path", type=str, help="Path to the model")
    args = parser.parse_args()

    print_model_summary(args.model_path)


if __name__ == "__main__":
    main()
