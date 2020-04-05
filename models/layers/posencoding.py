import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    """Implement the PE function.
    Note:
        WARN: dimension is [B, S, D]
        B is batch size
        S is sequence size
        D is hidden size
        In PyTorch, they wanna use [S, B, D], (TF users use [B, S, D])
    """

    def __init__(self, d_model: int, max_len: int = 5000,
                 dropout_rate: float = 0.1):
        super(PositionalEncoding, self).__init__()
        # PyTorch researcher uses dropout ... (Residual Dropout)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.size(1), :].clone().detach()
        return self.dropout(x)
