from pydantic import BaseModel
import torch
import yaml
from src.module import Tacotron
from src.symbols import txt2seq
from src.utils import AudioProcessor
import numpy as np
import tempfile

def load_ckpt(config, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = Tacotron(**config['model']['tacotron'])
    model.load_state_dict(ckpt['state_dict'])
    # This yeilds the best performance, not sure why
    # model.mel_decoder.eval()
    model.encoder.eval()
    model.postnet.eval()
    return model


class SpeechInput(BaseModel):
    text: str


class TtsModel:
    def __init__(self):
        self.config = None
        self.model = None

    def load_model(self):
        print("loading model")

        self.config = yaml.load(open('config/config.yaml', 'r'))
        self.model = load_ckpt(self.config, 'ckpt/checkpoint_step138000.pth')

        print("loaded")

    def generate_speech(self, input: SpeechInput) -> str:
        if not self.model:
            raise RuntimeError("Model files are not found!")

        seq = np.asarray(txt2seq(input.text))
        seq = torch.from_numpy(seq).unsqueeze(0)
        # Decode
        with torch.no_grad():
            mel, spec, attn = self.model(seq)
        # Generate wav file
        ap = AudioProcessor(**self.config['audio'])
        wav = ap.inv_spectrogram(spec[0].numpy().T)

        audio_path = f"generated_audio.wav"
        ap.save_wav(wav, audio_path)

        return audio_path
