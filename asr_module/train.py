# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import torch
    from data_setup import data_setup
    from engine import model_builder
    from engine import engine
    from utils import utils
    import torch.nn as nn
    import torchaudio
    import os

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")

    # Get an arg for num_epochs
    parser.add_argument("--num_epochs",
                        default=10,
                        type=int,
                        help="the number of epochs to train for")

    # Get an arg for batch_size
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="number of samples per batch")

    # Get an arg for learning_rate
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="learning rate to use for model")

    # Create an arg for training directory
    parser.add_argument("--train_data_url",
                        default="train-clean-100",
                        type=str,
                        help="directory file path to training data in standard image classification format")

    # Create an arg for test directory
    parser.add_argument("--test_data_url",
                        default="test-clean",
                        type=str,
                        help="directory file path to testing data in standard image classification format")

    # Create an arg for test directory
    parser.add_argument("--cache_dir",
                        default="./data/",
                        type=str,
                        help="directory file path to testing data in standard image classification format")

    # Get our arguments from the parser
    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    print(
        f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} and a learning rate of {LEARNING_RATE}")

    # Setup data urls
    train_data_url = args.train_data_url
    test_data_url = args.test_data_url
    cache_dir = args.cache_dir

    if not cache_dir:
        cache_dir = os.path.join(os.getcwd(), "data")

    print(f"[INFO] Training data url: {train_data_url}")
    print(f"[INFO] Testing data url: {test_data_url}")

    # Setup target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_audio_transforms=train_audio_transforms,
        valid_audio_transforms=valid_audio_transforms,
        batch_size=BATCH_SIZE,
        train_data_url=train_data_url,
        test_data_url=test_data_url,
        cache_dir=cache_dir
    )

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS
    }

    # Create model with help from model_builder.py
    model = model_builder.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    # Set loss and optimizer
    loss_fn = nn.CTCLoss(blank=28).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=NUM_EPOCHS,
                                                    anneal_strategy='linear')

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="asr_model_{}.pth".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')))
