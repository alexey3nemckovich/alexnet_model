import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext import data
import spacy


# Define the encoder architecture
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


# Define the decoder architecture
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)


# Define the seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


# Read and preprocess the data
data_path = 'kaggle/input/chatbot dataset.txt'


def preprocess_data(data_path):
    input_texts = []
    target_texts = []
    with open(data_path) as f:
        lines = f.read().split('\n')
    for line in lines[: min(600, len(lines) - 1)]:
        input_text = line.split('\t')[0]
        target_text = line.split('\t')[1]
        input_texts.append(input_text)
        target_texts.append(target_text)

    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = data.Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = data.Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    fields = [('src', SRC), ('trg', TRG)]
    examples = []
    for i in range(len(input_texts)):
        src = input_texts[i]
        trg = target_texts[i]
        examples.append(data.Example.fromlist([src, trg], fields))

    dataset = data.Dataset(examples, fields)

    SRC.build_vocab(dataset, min_freq=2)
    TRG.build_vocab(dataset, min_freq=2)

    train_data, val_data = dataset.split(split_ratio=0.8)
    train_iterator, val_iterator = data.BucketIterator.splits((train_data, val_data), batch_size=batch_size)

    return train_iterator, val_iterator, SRC, TRG


train_iterator, val_iterator, SRC, TRG = preprocess_data(data_path)

# Initialize model and optimizer
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)
enc_emb_dim = 256
dec_emb_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
enc_dropout = 0.5
dec_dropout = 0.5
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
attn = Attention(enc_hid_dim, dec_hid_dim)
dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

model = Seq2Seq(enc, dec, device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Define training parameters
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_iterator:
        src, trg = batch.src, batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_iterator)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_iterator:
            src, trg = batch.src, batch.trg
            output = model(src, trg, 0)  # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_iterator)

    print(f"Epoch: {epoch + 1:02}\nTrain Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")


# Inference
def translate_sentence(sentence, SRC, TRG, model, device, max_length=50):
    model.eval()
    tokens = sentence.lower().split()
    tokens = [SRC.init_token] + tokens + [SRC.eos_token]
    src_indexes = [SRC.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)])
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    mask = model.create_mask(src_tensor)
    trg_indexes = [TRG.vocab.stoi[TRG.init_token]]
    attentions = torch.zeros(max_length, 1, len(src_indexes)).to(device)
    for i in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == TRG.vocab.stoi[TRG.eos_token]:
            break
    trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


input_text = "Hello, how are you?"
translated_text, attentions = translate_sentence(input_text, SRC, TRG, model, device)
print("Input:", input_text)
print("Translated:", " ".join(translated_text))

# Unresolved reference 'output_word_dict_inv'
