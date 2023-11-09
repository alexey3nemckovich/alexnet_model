import random
import torch.utils.data
from utils import utils
from datetime import datetime
from tqdm import tqdm


def train_step(
        epoch: int,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        teacher_forcing_ratio: float,
        device: torch.device):

    train_loss = 0

    for input_tensor, target_tensor in dataloader:
        input_tensor = input_tensor.squeeze(0).to(device)
        target_tensor = target_tensor.squeeze(0).to(device)
        target_length = target_tensor.size(0)

        optimizer.zero_grad()

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        loss = model(input_tensor, target_tensor, use_teacher_forcing, loss_fn, device)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() / target_length

    return train_loss/ len(dataloader)


def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        teacher_forcing_ratio: float,
        target_dir: str,
        save_ratio: int,
        device: torch.device):

    for epoch in tqdm(range(1, epochs + 1), desc='Epochs', unit='epoch'):
        train_loss = train_step(epoch, model, dataloader, loss_fn, optimizer, teacher_forcing_ratio, device)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | "
                f"train_loss: {train_loss:.4f}"
            )
        
        if epoch % save_ratio == 0:
            file_name = "model_{}_epoch_{}.pth".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S'), epoch)
            print(f'[INFO] Saving model state to: {file_name}')

            utils.save_model(model=model,
                             target_dir=target_dir,
                             model_name="model_{}_epoch_{}.pth".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S'), epoch))
            
    file_name = "model_{}.pth".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S'))

    print(f'[INFO] Saving final model state to: {file_name}')

    utils.save_model(model=model,
                     target_dir=target_dir,
                     model_name="model_{}_epoch_{}.pth".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S'), epoch))
