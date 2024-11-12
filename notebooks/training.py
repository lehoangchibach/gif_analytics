import json
import gc
import os
import pickle as pk
from typing import List
import argparse
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary

# CONSTANTS
NUM_FRAMES = 5
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 6

def input_processor(video_path: str) -> np.ndarray:
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

    video_data = np.stack(frames) / 255.0
    return video_data.astype(np.float32)


class VideoDataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        template_ids: List[int],
        custom_processor: callable,
    ):
        self.video_paths = video_paths
        # Convert template IDs to tensor immediately
        self.template_ids = torch.tensor(template_ids, dtype=torch.long)
        self.custom_processor = custom_processor

        # Store unique template IDs for reference
        self.unique_templates = torch.unique(self.template_ids)
        self.num_classes = len(self.unique_templates)

        # Create mapping from template ID to class index (0 to num_classes-1)
        self.template_to_idx = {
            int(tid): idx for idx, tid in enumerate(self.unique_templates)
        }

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        template_id = int(self.template_ids[idx])

        video = self.custom_processor(video_path)
        video_tensor = torch.from_numpy(video)

        # Convert template ID to class index
        class_idx = self.template_to_idx[template_id]

        return video_tensor, torch.tensor(class_idx, dtype=torch.long)

    def get_original_template_id(self, class_idx: int) -> int:
        """Convert back from class index to original template ID"""
        return int(self.unique_templates[class_idx])


class VideoCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2, padding=1),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2, padding=1),

            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool3d(kernel_size=2, padding=1),
        )

        # Calculate the size of the flattened features
        self.flat_features = self._calculate_flat_features()

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 4096),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def _calculate_flat_features(self) -> int:
        x = torch.randn(1, 3, NUM_FRAMES, *IMAGE_SIZE)
        # x = torch.randn(1, NUM_FRAMES, 512, 512, 3)
        x = self.conv3d(x)
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_frames, height, width, channels)
        # Need to permute to: (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv3d(x)
        # print(f"Shape after conv3d: {x.shape}")

        x = x.view(x.size(0), -1)
        # print(f"Shape after flatten: {x.shape}")

        x = self.classifier(x)
        return x


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
    train_dataset,
    val_dataset,
    test_dataset,
    num_classes: int,
    num_epochs=1,
    is_summary=False,
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
            summary(model, (NUM_FRAMES, *IMAGE_SIZE, 3 ), batch_size=-1, device='cuda')
        exit()

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%\n"
            )

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model, f"./models/best_model_acc_{int(best_val_accuracy)}.pth")

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

    cleanup_ddp()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training categorized...")
    parser.add_argument('-dataset', type=str, help="Dataset name")
    parser.add_argument('-epochs', type=int, help="# Epochs")
    parser.add_argument('-summary', type=bool, help="Model summary")

    args = parser.parse_args()

    dataset = args.dataset
    if not dataset:
        data_path = "./data/"
        files = os.listdir(data_path)
        for i, file in enumerate(files):
            print(f"{i}: {file}")
        
        data_idx = int(input("Select file: "))
        dataset = f"{data_path}{files[data_idx]}"

    epochs = args.epochs
    if not epochs:
        epochs = int(input("Epochs: "))
    
    gc.collect()
    torch.cuda.empty_cache()

    with open(dataset, "rb") as f:
        X_data, Y_data = pk.load(f)

    dataset = VideoDataset(X_data, Y_data, custom_processor=input_processor)
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
        args=(world_size, train_dataset, val_dataset, test_dataset, num_classes, epochs, args.summary),
        nprocs=world_size,
        join=True,
    )
