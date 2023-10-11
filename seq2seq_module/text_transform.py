import torch
from special_symbols import BOS_IDX, EOS_IDX
from string_utils import normalize_string
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from special_symbols import *
from file_utils import read_pairs


def yield_tokens(data_iter, token_transform):
    for data_sample in data_iter:
        yield token_transform(data_sample)


def truncate_func(max_length):
    def truncate_to_max_length(token_ids):
        return token_ids[:max_length]

    return truncate_to_max_length


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))).unsqueeze(1)


def build_tokenizer():
    return get_tokenizer('spacy', language='en_core_web_sm')


def preprocess_data(file_path, max_length):
    pairs = read_pairs(file_path)

    # convert to list
    question = [pair[0] for pair in pairs]
    answer = [pair[1] for pair in pairs]
    words = question + answer

    # build vocab & transform
    token_transform = build_tokenizer()

    vocab = build_vocab_from_iterator(
        yield_tokens(words, token_transform),
        min_freq=1,
        specials=SPECIAL_SYMBOLS,
        special_first=True)

    # set default index
    vocab.set_default_index(UNK_IDX)
    
    return pairs, vocab


def build_text_transform(vocab, max_length):
    token_transform = build_tokenizer()

    text_transform = sequential_transforms(
        token_transform,
        vocab,
        tensor_transform,
        truncate_func(max_length))
    
    return text_transform


def text_encode(s, text_transform):
    return text_transform(normalize_string(s))


def text_decode(tokens, vocab):
    words_tokens = tokens

    words_tokens.remove(BOS_IDX)
    words_tokens.remove(EOS_IDX)

    words = vocab.lookup_tokens(words_tokens)
    
    return ' '.join(words)
