from pydantic import BaseModel
from engine.model_builder import Seq2SeqModel
from text_processing.text_transform import build_text_transform, text_decode, text_encode
from utils.utils import load_model
import torch


class ResponseInput(BaseModel):
    input: str


class ResponseOutput(BaseModel):
    output: str


class ChatbotModel:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.text_transform = None
        self.device = None

    def load_model(self):
        print("loading model")
        model_file_path = 'models/model.pth'
        vocab_file_path = 'models/vocab.pt'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = torch.load(vocab_file_path, map_location='cpu')

        model = Seq2SeqModel(len(self.vocab), 512, 100).to(self.device)
        load_model(model, model_file_path)

        self.model = model.to(self.device)
        self.text_transform = build_text_transform(self.vocab, model.max_length)

        print("loaded")

    def generate_response(self, input: ResponseInput) -> ResponseOutput:
        if not self.model or not self.vocab:
            raise RuntimeError("Model files are not found!")
        output = text_decode(self.model.forward_inference(text_encode(input.input, self.text_transform).to(self.device), self.device), self.vocab)
        return ResponseOutput(output=output)
