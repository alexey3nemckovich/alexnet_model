import torch
import torch.nn.functional as F
import torch.utils.data

from text_processing import text_transform
from utils.metrics import cer, wer


def greedy_decoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler,
               device: torch.device,
               epoch):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    model.train()
    data_len = len(dataloader.dataset)

    train_loss = 0
    train_cer, train_wer = [], []
    for batch_idx, _data in enumerate(dataloader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = loss_fn(output, labels, input_lengths, label_lengths)
        train_loss += loss.item()

        loss.backward()

        optimizer.step()
        scheduler.step()

        decoded_preds, decoded_targets = greedy_decoder(output.transpose(0, 1), labels, label_lengths)
        for j in range(len(decoded_preds)):
            train_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            train_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / len(dataloader), loss.item()))

    train_loss = train_loss / len(dataloader)
    train_cer = sum(train_cer) / len(train_cer)
    train_wer = sum(train_wer) / len(train_wer)
    return train_loss, train_cer, train_wer


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    model.eval()

    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(dataloader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = loss_fn(output, labels, input_lengths, label_lengths)
            test_loss += loss.item()

            decoded_preds, decoded_targets = greedy_decoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    test_loss = test_loss / len(dataloader)
    test_cer = sum(test_cer) / len(test_cer)
    test_wer = sum(test_wer) / len(test_wer)
    return test_loss, test_cer, test_wer


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_cer": [],
        "train_wer": [],
        "test_loss": [],
        "test_cer": [],
        "test_wer": [],
    }

    print("Launching model training...")

    for epoch in range(1, epochs+1):
        train_loss, train_cer, train_wer = train_step(model=model,
                                                      dataloader=train_dataloader,
                                                      loss_fn=loss_fn,
                                                      optimizer=optimizer,
                                                      scheduler=scheduler,
                                                      device=device,
                                                      epoch=epoch)

        test_loss, test_cer, test_wer = test_step(model=model,
                                                  dataloader=test_dataloader,
                                                  loss_fn=loss_fn,
                                                  device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_cer: {train_cer:.4f} | "
            f"train_wer: {train_wer:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_cer: {test_cer:.4f} | "
            f"test_wer: {test_wer:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_cer"].append(train_cer)
        results["train_wer"].append(train_wer)
        results["test_loss"].append(test_loss)
        results["test_cer"].append(test_cer)
        results["test_wer"].append(test_wer)

    print("Model training was successfully completed")

    return results
