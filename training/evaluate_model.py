import argparse
import gc
import json
import os
import pickle as pk
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from constants import *
from dataset import VideoDataset
from helpers import load_checkpoint, video_processor
from model import VideoCNN


def setup_ddp(rank, world_size):
    """
    Setup for distributed evaluation
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """
    Clean up distributed evaluation
    """
    dist.destroy_process_group()


def evaluate_model(
    rank,
    world_size: int,
    dataset_name: str,
    test_dataset,
    num_classes: int,
    checkpoint_path: str,
):
    """
    Evaluate a pre-trained model on test dataset
    """
    # Setup DDP
    setup_ddp(rank, world_size)

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
    )

    # Create model and move it to GPU
    model = VideoCNN(num_classes=num_classes).to(rank)
    model = DDP(model, device_ids=[rank])

    # Load checkpoint
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load checkpoint
    start_epoch, val_accuracy, best_val_accuracy = load_checkpoint(
        rank, model, optimizer, checkpoint_path
    )

    if rank == 0:
        print(f"Loaded model trained up to epoch {start_epoch}")
        print(f"Previous validation accuracy: {val_accuracy:.2f}%")
        print(f"Previous best validation accuracy: {best_val_accuracy:.2f}%")

    # Perform evaluation
    test_loss, test_accuracy, test_report, conf_matrix = evaluate_model_ddp(
        model, test_loader, criterion, rank, "test"
    )

    if rank == 0:
        print("\nFinal evaluation on test set...")
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print("\nClassification Report:")
        print(test_report)
        print("\nConfusion Matrix:")

        # Save confusion matrix
        os.makedirs("./outputs", exist_ok=True)
        with open(f"./outputs/{dataset_name}_conf_matrix.json", "w") as f:
            json.dump(conf_matrix.tolist(), f)

        print(conf_matrix)

    cleanup_ddp()


def evaluate_model_ddp(model, dataloader, criterion, device, phase="test"):
    """
    Distributed evaluation of model
    """
    model.eval()
    total_loss = torch.zeros(1).to(device)
    correct = torch.zeros(1).to(device)
    total = torch.zeros(1).to(device)
    predictions = []
    true_labels = []

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.cuda(device, non_blocking=True)
            labels = labels.cuda(device, non_blocking=True)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            del videos, labels, outputs
            torch.cuda.empty_cache()

    # Synchronize metrics across all GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    # Calculate metrics
    avg_loss = (total_loss / len(dataloader)).item()
    accuracy = (100.0 * correct / total).item()

    # Gather all predictions and labels
    all_predictions = [[] for _ in range(dist.get_world_size())]
    all_labels = [[] for _ in range(dist.get_world_size())]

    dist.all_gather_object(all_predictions, predictions)
    dist.all_gather_object(all_labels, true_labels)

    # Process metrics on main process
    predictions = [p for sublist in all_predictions for p in sublist]
    true_labels = [l for sublist in all_labels for l in sublist]
    report = classification_report(true_labels, predictions, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)

    return avg_loss, accuracy, report, conf_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("-dataset", type=str, help="Dataset file path")
    parser.add_argument(
        "-checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    args = parser.parse_args()

    # Dataset selection
    dataset = args.dataset
    if not dataset:
        data_path = "./data/"
        files = os.listdir(data_path)
        for i, file in enumerate(files):
            print(f"{i}: {file}")

        data_idx = int(input("Select file: "))
        dataset = f"{data_path}{files[data_idx]}"

    # Model checkpoint validation
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist.")
        sys.exit(1)

    # Cleanup and cache clearing
    gc.collect()
    torch.cuda.empty_cache()

    # Load dataset
    with open(dataset, "rb") as f:
        X_data, Y_data = pk.load(f)

    dataset_name = dataset.split("/")[-1].split(".")[0]
    video_dataset = VideoDataset(X_data, Y_data, custom_processor=video_processor)
    num_classes = video_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    # Create train, validation, and test datasets
    train_val_indices, test_indices = train_test_split(
        list(range(len(video_dataset))), test_size=0.2, random_state=42
    )

    test_dataset = Subset(video_dataset, test_indices)

    print(f"Test dataset size: {len(test_dataset)} samples")

    # Multi-processing setup
    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        evaluate_model,
        args=(
            world_size,
            dataset_name,
            test_dataset,
            num_classes,
            args.checkpoint,
        ),
        nprocs=world_size,
        join=True,
    )
