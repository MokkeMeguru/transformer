import numpy as np
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, pad_idx):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.d_model)
