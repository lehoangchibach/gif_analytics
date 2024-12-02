from typing import List

import torch
from numpy import ndarray
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    video_paths: list[str]
    template_ids: ndarray[int]
    custom_processor: callable
    unique_templates: ndarray[int]
    num_classes: int
    template_to_idx: dict
    repetitions: int

    def __init__(
        self,
        video_paths: List[str],
        template_ids: List[int],
        custom_processor: callable,
        repetitions: int = 1,
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

        self.repetitions = repetitions

    def __len__(self):
        return len(self.video_paths) * self.repetitions

    def __getitem__(self, idx):
        # Get original indexes
        idx = idx // self.repetitions

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
