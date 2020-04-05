import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logging import getLogger
logger = getLogger(__name__)

from .multiheadattn import MultiheadAttention
from .positionwiseffn import PositionwiseFeedForward
from .embedding import Embedding

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_head: int,
                 dropout_rate: float=0.1):
        super(EncoderLayer, self).__init__()
        assert d_model % num_head == 0
        d_k = d_model // num_head
        d_v = d_model // num_head

        self.slf_attn = MultiheadAttention(
            num_head, d_model, d_k, d_v, dropout_rate=dropout_rate)

        self.ffn = PositionwiseFeedForward(
            d_model, d_ff, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x, attn = self.slf_attn(x, x, x, mask=mask)
        x = self.ffn(x)
        return x, attn

class Encoder(nn.Module):
    def __init__(self,
                 max_length: int,
                 num_layer: int,
                 num_head: int,
                 d_model: int,
                 d_ff: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.max_length= max_length
        self.d_model = d_model

        self.layer_stack = nn.ModuleList ([
            EncoderLayer(d_model, d_ff, num_head, dropout_rate)
            for _ in range(num_layer)])

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        out = src
        attns = []
        for layer in self.layer_stack:
            out, attn = layer(out, mask=mask)
            attns += [attn]
        return out, attns

