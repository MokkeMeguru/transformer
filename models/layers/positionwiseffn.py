"""
Author: "Yu-Hsiang Huang"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logging import getLogger
logger = getLogger(__name__)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout_rate:float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layer_norm(x)
        x =  self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x
