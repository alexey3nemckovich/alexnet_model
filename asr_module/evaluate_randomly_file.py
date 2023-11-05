# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing

from text_processing import text_transform



def greedy_decoder_one(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
        #print(decodes)
    return decodes


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
    import torch.nn.functional as F

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")

    # Get an arg for num_epochs
    parser.add_argument("--n",
                        default=100,
                        type=int,
                        help="the number of epochs to train for")

    parser.add_argument("--model_file_path",
                        # default="./chatbot dataset.txt",
                        default='/home/alex/projects/ml/ml_final_project/asr_module/models/asr_model_2023_09_02_21_58_53.pth',
                        type=str,
                        help="directory file path to training data in standard image classification format")

    # Get an arg for batch_size
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="number of samples per batch")

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
    N = args.n
    BATCH_SIZE = args.batch_size
    MODEL_FILE_PATH = args.model_file_path
    print(
        f"[INFO] Evaluating a model for {N} samples with batch size {BATCH_SIZE}")

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

    audio_transforms = torchaudio.transforms.MelSpectrogram()

    # Create DataLoaders with help from data_setup.py
    _, test_dataloader = data_setup.create_dataloaders(
        train_audio_transforms=audio_transforms,
        valid_audio_transforms=audio_transforms,
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
    }

    # Create model with help from model_builder.py
    model = model_builder.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    )

    utils.load_model(model, MODEL_FILE_PATH)

    spectrograms = []

    waveform, sample_rate = torchaudio.load('/home/alex/projects/ml/ml_final_project/asr_module/data/LibriSpeech/train-clean-100/19/198/19-198-0001.flac')

    spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
    spectrograms.append(spec)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    print(spectrograms.shape)

    output = model(spectrograms)  # (batch, time, n_class)
    print(output.shape)
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)  # (time, batch, n_class)
    print(output.shape)

    #print(output)
    decodes = greedy_decoder_one(output.transpose(0, 1))
    print(decodes)



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