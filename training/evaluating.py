import argparse
import gc
import json
import os
import pickle as pk
import sys

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from constants import *
from dataset import VideoDataset
from helpers import (
    cleanup_ddp,
    evaluate_model_ddp,
    load_checkpoint,
    setup_ddp,
    test_video_processor,
)


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

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
        sampler=test_sampler,
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
    video_dataset = VideoDataset(X_data, Y_data, custom_processor=test_video_processor)
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
