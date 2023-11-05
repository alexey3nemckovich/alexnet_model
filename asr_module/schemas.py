from fastapi import UploadFile
from pydantic import BaseModel
from engine.model_builder import SpeechRecognitionModel
from utils.utils import load_model
import torch
import torch.nn.functional as F
import torchaudio
from text_processing import text_transform


# class ConvertInput(BaseModel):
#     input: File


class TranscriptionOutput(BaseModel):
    output: str

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


class AsrModel:
    def __init__(self):
        #self.device = None
        self.text_transform = None
        self.model = None
        self.audio_transforms = None

    def load_model(self):
        print("loading model")
        model_file_path = 'models/asr_model_2023_09_02_21_58_53.pth'

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model = SpeechRecognitionModel(
            hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        )#.to(self.device)

        load_model(model, model_file_path)

        self.model = model#.to(self.device)
        self.audio_transforms = torchaudio.transforms.MelSpectrogram()

        print("loaded")

    def get_audio_transcription(self, file: UploadFile) -> TranscriptionOutput:
        if not self.model:
            raise RuntimeError("Model files are not found!")

        waveform, sample_rate = torchaudio.load(file.file)

        spectrograms = []

        spec = self.audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

        output = self.model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        decodes = greedy_decoder_one(output.transpose(0, 1))

        return TranscriptionOutput(output=decodes[0])
