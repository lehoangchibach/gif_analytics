import cv2
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from constants import *


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


def video_processor(video_path: str) -> np.ndarray:
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

    # seq = va.Sequential(
    #     [
    #         va.Pepper(),
    #         va.RandomShear(x=0.05, y=0.05),
    #         va.RandomTranslate(x=25, y=25),
    #         va.RandomRotate(20),
    #         va.Sometimes(0.5, va.HorizontalFlip()),
    #     ]
    # )

    # frames = seq(frames)
    video_data = np.stack(frames)
    return video_data.astype(np.float32)
