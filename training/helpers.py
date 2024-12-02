import os

import cv2
import numpy as np
import torch
import torch.distributed as dist
import vidaug.augmentors as va  # noqa
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.parallel import DistributedDataParallel as DDP

from constants import *


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


def train_video_processor(video_path: str) -> np.ndarray:
    frames = []
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError("Could not read frames from video file")

    # Calculate frame indices to extract
    frame_indices = [
        i * (total_frames - 1) // (NUM_FRAMES - 1) for i in range(NUM_FRAMES)
    ]

    for frame_idx in frame_indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, IMAGE_SIZE)

            frames.append(frame_rgb)

    # Release video capture object
    cap.release()

    while len(frames) < 5:
        frames.append(frames[-1])

    seq = va.Sequential(
        [
            va.Pepper(),
            va.RandomShear(x=0.05, y=0.05),
            va.RandomTranslate(x=25, y=25),
            va.RandomRotate(20),
            va.Sometimes(0.5, va.HorizontalFlip()),
        ]
    )

    frames = seq(frames)
    video_data = np.stack(frames)
    return video_data.astype(np.float32)


def test_video_processor(video_path: str) -> np.ndarray:
    frames = []
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError("Could not read frames from video file")

    # Calculate frame indices to extract
    frame_indices = [
        i * (total_frames - 1) // (NUM_FRAMES - 1) for i in range(NUM_FRAMES)
    ]

    for frame_idx in frame_indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, IMAGE_SIZE)

            frames.append(frame_rgb)

    # Release video capture object
    cap.release()

    while len(frames) < 5:
        frames.append(frames[-1])

    video_data = np.stack(frames)
    return video_data.astype(np.float32)


def save_checkpoint(model, optimizer, epoch, val_accuracy, best_val_accuracy, filename):
    """
    Save model checkpoint with all necessary information to resume training
    """
    # Get model state dict depending on whether it's DDP or not
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
        "best_val_accuracy": best_val_accuracy,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(rank, model, optimizer, filename):
    """
    Load checkpoint and return necessary training state information
    """
    # Load checkpoint on CPU first to avoid GPU RAM spike
    checkpoint = torch.load(filename, map_location="cpu")

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move optimizer state to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(rank)

    return (
        checkpoint["epoch"],
        checkpoint["val_accuracy"],
        checkpoint["best_val_accuracy"],
    )


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
