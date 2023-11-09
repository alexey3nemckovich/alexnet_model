import os
from pathlib import Path

import torch.nn
import pickle


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    # print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def load_model(model: torch.nn.Module,
               file_path: str):
  model_state_dict = torch.load(file_path, map_location='cpu')

  model.load_state_dict(model_state_dict)


def save_transform(obj, dir, file_name):
    os.makedirs(dir, exist_ok=True)

    full_file_path = "{}/{}".format(dir, file_name)

    with open(full_file_path, 'wb') as file:
      pickle.dump(obj, file)


def load_transform(dir, file_name):
    os.makedirs(dir, exist_ok=True)

    full_file_path = "{}/{}".format(dir, file_name)
    
    with open(full_file_path, 'rb') as file:
      pickle.load(file)
