import argparse
import torch
import pickle as pk
from model import VideoCNN
from dataset import VideoDataset
from helpers import test_video_processor

def predict_example(model_path, dataset_path: str, example_path: str):
    """
    Load a pre-trained model and predict the class for a given example
    """
    # Load the dataset to get number of classes
    with open(dataset_path, "rb") as f:
        X_data, Y_data = pk.load(f)

    # Create dataset to determine number of classes
    video_dataset = VideoDataset(X_data, Y_data, custom_processor=test_video_processor)
    num_classes = video_dataset.num_classes

    # Create model
    model = VideoCNN(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Add batch dimension
    example_video = torch.from_numpy(test_video_processor(example_path)).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(example_video)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    # Print results
    print(f"Model Path: {model_path}")
    print(f"Example Path: {example_path}")
    template_id = video_dataset.get_original_template_id(predicted_class)
    print(f"Predicted Label: {template_id}")

    return template_id

def main():
    parser = argparse.ArgumentParser(description="Model Prediction Script")
    parser.add_argument("model", type=str, help="Path to model checkpoint")
    parser.add_argument("dataset", type=str, help="Path to original dataset")
    parser.add_argument("example", type=str, help="Path to predicting video")
    
    args = parser.parse_args()
    
    predict_example(args.model, args.dataset, args.example)

if __name__ == "__main__":
    main()


