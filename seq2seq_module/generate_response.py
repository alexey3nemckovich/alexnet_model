from model_builder import Seq2SeqModel
from utils import load_model
from text_transform import build_text_transform
from file_utils import read_pairs
from evaluation_utils import evaluateRandomly
from text_transform import text_encode, text_decode
import torch


def generate_response(request: str, model_file_path: str, vocab_file_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = torch.load(vocab_file_path)

    model = Seq2SeqModel(len(vocab), 512, 100).to(device)

    load_model(model, model_file_path)

    model = model.to(device)

    text_transform = build_text_transform(vocab, model.max_length)

    return text_decode(model.forward_inference(text_encode(request, text_transform).to(device), device), vocab)
