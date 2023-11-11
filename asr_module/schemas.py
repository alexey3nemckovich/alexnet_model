import torch
import torchaudio
from fastapi import UploadFile
from pydantic import BaseModel
from decoder.greedy_decoder import GreedyCTCDecoder


# class ConvertInput(BaseModel):
#     input: File


class TranscriptionOutput(BaseModel):
    text: str


class AsrModel:
    def __init__(self):
        self.device = None
        self.bundle = None
        self.model = None

    def load_model(self):
        print("loading model")

        torch.random.manual_seed(0)
        torch.hub.set_dir("models")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
        self.model = self.bundle.get_model().to(self.device)

        print("loaded")

    def get_audio_transcription(self, file: UploadFile) -> TranscriptionOutput:
        if not self.model:
            raise RuntimeError("Model files are not found!")

        waveform, sample_rate = torchaudio.load(file.file)
        waveform = waveform.to(self.device)

        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

        with torch.inference_mode():
            emission, _ = self.model(waveform)

        decoder = GreedyCTCDecoder(labels=self.bundle.get_labels())
        transcript = decoder(emission[0]).replace("|", " ").lower()

        return TranscriptionOutput(text=transcript)
