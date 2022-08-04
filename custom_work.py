import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import json

from transformers import BartTokenizer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    """
    ntoken = size of vocab
    d_model = model dimension (used internally; embeddings project to this space)
    nhead = number of attention heads
    d_hid = dimention of feed forward networks
    nlayers = number of transformer encoder/decoder layers

    """
    def __init__(self, ntoken: int, nimagetoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, activation: str = 'gelu'):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, activation=activation)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

        self.decoder_embedding = nn.Embedding(nimagetoken, d_model)
        self.decoder = nn.Linear(d_model, nimagetoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, output, tgt_mask=tgt_mask)
        return self.decoder(output)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if __name__ == '__main__':
    # download_all_used_models()
    tokenizer = BartTokenizer.from_pretrained("model-downloads/bart-tokenizer")

    example = "There is no just cause for an invasion of Iraq."
    token_dict = tokenizer(example)
    tokens = torch.tensor(token_dict['input_ids'])
    src_mask = None


    # Model params match BART except for the nimagetoken, which matches the dalle dVAE
    model = TransformerModel(ntoken=50265, nimagetoken=8192, d_model=1024, nhead=16, d_hid=12,
                    nlayers=12, dropout=0.1)

    tgt = torch.zeros((13,)).int()
    tgt_mask = torch.triu(torch.ones(13, 13) * float('-inf'), diagonal=1)  # Causal mask
    out = model(tokens, tgt, src_mask, tgt_mask)

    print(out.shape)