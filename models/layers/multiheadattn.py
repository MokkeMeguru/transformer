"""
Author: "Yu-Hsiang Huang"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logging import getLogger
from .scaled_dot_prod_attention import ScaledDotProdAttention
# from chap3.models.layers.scaled_dot_prod_attention import ScaledDotProdAttention
logger = getLogger(__name__)


class MultiheadAttention(nn.Module):
    """multi head attention module
    Examples:
        >>> ma = MultiheadAttention(num_head, d_model, d_k, d_v)
        >>> out, attn = ma(query, key, value, mask=mask)
    Notes:
        Qs, Ks, Vs <- Q, K, V
        out = Linear(concat(scaled_dot_prod_attention(Qi, Ki, Vi)))
        where.
           Q (query): [B, S , d_model]
           K   (key): [B, S', d_model]
           V (value): [B, S', d_model]
           mask     : [B, S', d_model]
           Qs (queries): num_head x [B, S, d_model // num_head]
           out      : [B, S, d_model]
           attn     : [B, num_head, S, S']
        when this layer used at source attention,
           S is target's sequence length
           S' is source's sequence length
    """
    def __init__(self, num_head, d_model, d_k, d_v, dropout_rate=0.1):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, num_head * d_k, bias = False)
        self.w_ks = nn.Linear(d_model, num_head * d_k, bias = False)
        self.w_vs = nn.Linear(d_model, num_head * d_v, bias = False)

        nn.init.xavier_normal_(self.w_qs.weight)
        nn.init.xavier_normal_(self.w_ks.weight)
        nn.init.xavier_normal_(self.w_vs.weight)

        self.fc = nn.Linear(num_head * d_v, d_model, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.sdpa = ScaledDotProdAttention(temparature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None):
        d_k, d_v, num_head = self.d_k, self.d_v, self.num_head
        B = query.size(0)

        residual = query
        query = self.layer_norm(query)

        query = self.w_qs(query).view(B, -1, num_head, d_k).transpose(1, 2)
        key = self.w_ks(key).view(B, -1, num_head, d_k).transpose(1, 2)
        value = self.w_vs(value).view(B, -1, num_head, d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.sdpa(query, key, value, mask)
        out = out.transpose(1, 2).contiguous().view(
            B, -1, self.d_model)
        out = self.dropout(self.fc(out))
        out += residual
        return out, attn

def main():
    batch_size= 64
    seq_len = 13
    d_model = 128
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    mask = torch.zeros([batch_size, 1, seq_len])
    mh = MultiheadAttention(8, d_model, d_model // 4, d_model // 4, 0.1)
    out = mh(query, key, value, mask)
    print(out[0].shape)
