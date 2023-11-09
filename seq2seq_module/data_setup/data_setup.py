"""
Contains functionality for creating PyTorch DataLoaders for
ASR data.
"""
import os

import torch
from torch.utils.data import Dataset

NUM_WORKERS = os.cpu_count()


class ConversationDataset(Dataset):
    def __init__(self, pairs, text_transform, init_dataset=True) -> None:
        super().__init__()

        if init_dataset:
            self._init_dataset(pairs, text_transform)

    def __len__(self):
        return len(self.src_batch)

    def __getitem__(self, idx):
        return self.src_batch[idx], self.tgt_batch[idx]

    def _init_dataset(self, pairs, text_transform):
        # transform batch
        self.src_batch, self.tgt_batch = [], []
        for src_sample, tgt_sample in pairs:
            self.src_batch.append(text_transform(src_sample))
            self.tgt_batch.append(text_transform(tgt_sample))

        # finish init dataset


def create_dataloader(
    pairs,
    text_transform,
    num_workers = NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      transform: torchvision transforms to perform on training and testing data.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: An integer for number of workers per DataLoader.
      train_url: Train database name.
      test_url: Test database name.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, test_dataloader, class_names = \
          = create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               transform=some_transform,
                               batch_size=32,
                               num_workers=4)
                               :param num_workers:
                               :param cache_dir:
                               :param train_data_url:
                               :param batch_size:
                               :param valid_audio_transforms:
                               :param train_audio_transforms:
                               :param test_data_url:
    """
    dataset = ConversationDataset(pairs, text_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
