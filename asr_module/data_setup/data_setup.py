"""
Contains functionality for creating PyTorch DataLoaders for
ASR data.
"""
import os

import torch
import torch.nn as nn
import torch.utils.data
import torchaudio

from text_processing import text_transform

NUM_WORKERS = os.cpu_count()


def data_processing(data, audio_transform: torch.nn.Module):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        spec = audio_transform(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def create_dataloaders(
        train_audio_transforms: torch.nn.Module,
        valid_audio_transforms: torch.nn.Module,
        batch_size: int,
        train_data_url: str = "train-clean-100",
        test_data_url: str = "test-clean",
        cache_dir: str = "./data/",
        num_workers: int = NUM_WORKERS,
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
    # Use ImageFolder to create dataset(s)
    print(f"[INFO] Loading train data...")
    train_data = torchaudio.datasets.LIBRISPEECH(cache_dir, url=train_data_url, download=True)
    print(f"[INFO] Train data was successfully loaded")

    print(f"[INFO] Loading test data...")
    test_data = torchaudio.datasets.LIBRISPEECH(cache_dir, url=test_data_url, download=True)
    print(f"[INFO] Test data was successfully loaded")

    # Turn images into data loaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda x: data_processing(x, train_audio_transforms),
                                                   num_workers=num_workers,
                                                   pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=lambda x: data_processing(x, valid_audio_transforms),
                                                  num_workers=num_workers,
                                                  pin_memory=True)

    return train_dataloader, test_dataloader
