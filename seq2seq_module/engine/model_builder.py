import torch.nn as nn
import torch.nn.functional as F
import torch
from text_processing.special_symbols import BOS_IDX, EOS_IDX


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Seq2SeqModel(nn.Module):
    def __init__(self) -> None:
        super(Seq2SeqModel, self).__init__()

    def __init__(self, vocabluary_size = 100, hidden_size = 512, max_length = 100) -> None:
        super(Seq2SeqModel, self).__init__()

        self.encoder = EncoderRNN(vocabluary_size, hidden_size)
        self.decoder = AttentionDecoder(hidden_size, vocabluary_size, max_length=max_length, dropout_p=0.1)

        self.max_length = max_length

    def encoder_forward(self, input_tensor, device):
        encoder_hidden = self.encoder.initHidden(device)

        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        return encoder_hidden, encoder_outputs

    def forward_inference(self, input_tensor, device):
        with torch.no_grad():
            encoder_hidden, encoder_outputs = self.encoder_forward(input_tensor, device)

            decoder_input = torch.tensor([[BOS_IDX]], device=device)
            decoder_hidden = encoder_hidden

            decoded_tokens = []

            for i in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoded_tokens.append(decoder_input)
                
                if decoder_input.item() == EOS_IDX:
                    break

            return decoded_tokens

    def forward(self, input_tensor, target_tensor, use_teacher_forcing, loss_fn, device):
        loss = 0

        encoder_hidden, encoder_outputs = self.encoder_forward(input_tensor, device)

        decoder_input = torch.tensor([[BOS_IDX]], device=device)
        decoder_hidden = encoder_hidden

        target_length = target_tensor.size(0)

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += loss_fn(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += loss_fn(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_IDX:
                    break

        return loss
