import argparse

import torch
from helpers import train_video_processor
from model import VideoCNN


def predict_example(model_path, example_path: str):
    """
    Load a pre-trained model and predict the class for a given example
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Create model
    num_classes = checkpoint["num_classes"]
    model = VideoCNN(num_classes=num_classes)

    # Load model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Add batch dimension
    example_video = torch.from_numpy(train_video_processor(example_path)).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(example_video)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    # Print results
    print(f"Model Path: {model_path}")
    print(f"Example Path: {example_path}")
    template_id = checkpoint["unique_templates"][predicted_class]
    print(f"Predicted Label: {template_id}")

    return template_id


def main():
    parser = argparse.ArgumentParser(description="Model Prediction Script")
    parser.add_argument("model", type=str, help="Path to model checkpoint")
    parser.add_argument("example", type=str, help="Path to predicting video")

    args = parser.parse_args()

    predict_example(args.model, args.example)


if __name__ == "__main__":
    main()
