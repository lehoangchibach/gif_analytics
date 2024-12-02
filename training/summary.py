import argparse
import pickle as pk
import torch
from torchsummary import summary

from constants import *
from model import VideoCNN
from dataset import VideoDataset


def print_model_summary(num_classes):
    """
    Load a saved PyTorch model and print its summary.

    Args:
        model_path (str): Path to the saved model checkpoint
        num_classes (int): Number of classes in the model
    """
    # Check if CUDA is available
    device = torch.device("cpu")

    # Recreate the model architecture
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
    parser.add_argument(
        "dataset", type=str, help="Path to the dataset"
    )
    args = parser.parse_args()

    with open(args.dataset, "rb") as f:
            X_data, Y_data = pk.load(f)
    dataset = VideoDataset(X_data, Y_data, custom_processor=None)
    num_classes = dataset.num_classes
    print(f"Nums of classes: {num_classes}")

    
    print_model_summary(num_classes)


if __name__ == "__main__":
    main()
