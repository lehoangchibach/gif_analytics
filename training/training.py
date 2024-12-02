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
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary

from constants import *
from dataset import VideoDataset
from helpers import load_checkpoint, save_checkpoint, video_processor
from model import VideoCNN


def setup_ddp(rank, world_size):
    """
    Setup for distributed training
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """
    Clean up distributed training
    """
    dist.destroy_process_group()


def evaluate_model_ddp(model, dataloader, criterion, device, phase="val"):
    """
    Evaluate model when DDP is already set up.
    Assumes model is already wrapped in DDP and process groups are initialized.
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

            if phase == "test":
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

    # For test phase, gather all predictions and labels
    if phase == "test":
        all_predictions = [[] for _ in range(dist.get_world_size())]
        all_labels = [[] for _ in range(dist.get_world_size())]

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_labels, true_labels)

        # Only process and return metrics on main process
        predictions = [p for sublist in all_predictions for p in sublist]
        true_labels = [l for sublist in all_labels for l in sublist]
        report = classification_report(true_labels, predictions, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)
        return avg_loss, accuracy, report, conf_matrix

    return avg_loss, accuracy


def train_model_ddp(
    rank,
    world_size: int,
    dataset_name: str,
    train_dataset,
    val_dataset,
    test_dataset,
    num_classes: int,
    num_epochs=1,
    is_summary=False,
    checkpoint_path=None,
    checkpoint_frequency=10,
):
    best_val_accuracy = -1

    # Setup DDP
    setup_ddp(rank, world_size)

    # Create dataloader
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        sampler=train_sampler,
        persistent_workers=True,
    )

    # Create validation and test dataloaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # Create model and move it to GPU
    model = VideoCNN(num_classes=num_classes).to(rank)
    model = DDP(model, device_ids=[rank])

    # SUMMARY
    if is_summary:
        if rank == 0:
            summary(model, (NUM_FRAMES, *IMAGE_SIZE, 3), batch_size=-1, device="cuda")
        exit()

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if checkpoint_path and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, val_accuracy, best_val_accuracy = load_checkpoint(
            rank, model, optimizer, checkpoint_path
        )
        if rank == 0:
            print(
                f"Resuming from epoch {start_epoch} with validation accuracy {val_accuracy:.2f}%"
            )

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0 and rank == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {100.0 * correct / total:.2f}%"
                )

        val_loss, val_accuracy = evaluate_model_ddp(
            model, val_loader, criterion, rank, "val"
        )
        if rank == 0:
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total

            print(f"\nVALIDATION...")
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"\nTrain Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2f}%, "
                f"\nVal Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%\n"
            )

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_file = f"./models/{dataset_name}/best_model_acc_{int(best_val_accuracy)}.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    val_accuracy,
                    best_val_accuracy,
                    best_model_file,
                )
                print(f"Saved best model to {best_model_file}")

            if (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_file = (
                    f"./models/{dataset_name}/checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    val_accuracy,
                    best_val_accuracy,
                    checkpoint_file,
                )
                print(f"Saved checkpoint to {checkpoint_file}")

            sys.stdout.flush()

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
        with open(f"./outputs/conf_matrix.json", "w") as f:
            json.dump(conf_matrix.tolist(), f)
        print(conf_matrix)

        # Save final model
        final_checkpoint = f"./models/{dataset_name}/final_model.pth"
        save_checkpoint(
            model,
            optimizer,
            num_epochs,
            val_accuracy,
            best_val_accuracy,
            final_checkpoint,
        )
        print(f"Saved final model to {final_checkpoint}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training categorized...")
    parser.add_argument("-dataset", type=str, help="Dataset name")
    parser.add_argument("-epochs", type=int, help="# Epochs")
    parser.add_argument("-summary", type=bool, help="Model summary")
    parser.add_argument(
        "-checkpoint", type=str, help="Path to checkpoint file to resume training"
    )
    parser.add_argument(
        "-checkpoint_freq",
        type=int,
        default=10,
        help="Checkpoint saving frequency (epochs)",
    )

    args = parser.parse_args()

    dataset = args.dataset
    if not dataset:
        data_path = "./data/"
        files = os.listdir(data_path)
        for i, file in enumerate(files):
            print(f"{i}: {file}")

        data_idx = int(input("Select file: "))
        dataset = f"{data_path}{files[data_idx]}"

    # Model save output
    dataset_name = dataset.split("/")[-1].split(".")[0]
    directory = f"./models/{dataset_name}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

    epochs = args.epochs or int(input("Epochs: "))

    gc.collect()
    torch.cuda.empty_cache()

    with open(dataset, "rb") as f:
        X_data, Y_data = pk.load(f)

    dataset = VideoDataset(X_data, Y_data, custom_processor=video_processor)
    num_classes = dataset.num_classes
    print(f"Nums of classes: {num_classes}")

    # Create train, validation, and test datasets
    train_val_indices, test_indices = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.25, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Dataset splits:")
    print(f"\tTraining samples: {len(train_dataset)}")
    print(f"\tValidation samples: {len(val_dataset)}")
    print(f"\tTest samples: {len(test_dataset)}")

    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        train_model_ddp,
        args=(
            world_size,
            dataset_name,
            train_dataset,
            val_dataset,
            test_dataset,
            num_classes,
            epochs,
            args.summary,
            args.checkpoint,
            args.checkpoint_freq,
        ),
        nprocs=world_size,
        join=True,
    )