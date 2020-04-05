"""
Author: "Yu-Hsiang Huang"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logging import getLogger
logger = getLogger(__name__)

class ScaledDotProdAttention(nn.Module):
    """scaled dot prod attention module
    Examples:
         >>> sdpa = ScaledDotProdAttention(temparature=d_k ** 0.5)
         >>> sdpa(query, key, value, mask=mask)
    Notes:
         Attn = softmax(Q K / sqrt(d_model))
         output = Attn V
         where.
           Q (query): [B, num_head, S , d_k]
           K   (key): [B, num_head, S', d_k]
           V (value): [B, num_head, S', d_k]
           mask     : [B,        1, S', d_k]
           Attn     : [B, num_head, S, S']
           B is batch size
           num_head is number of heads of multi-head-attention
           S, S' sequence length
           d_k is d_model // num_head
         when this layer used at source attention,
           S is target's sequence length
           S' is source's sequence length
    """
    def __init__(self, temparature: float, dropout_rate: float=0.1):
        super().__init__()
        self.temparature = temparature
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor= None):
        """
        Args:
            query: [B, num_head, S, d_k]
            key  : [B, num_head, S', d_k]
            value: [B, num_head, S', d_k]
        Returns:
            output: [B, S, len_q, len_k]
        Notes:
            where d_k = d_model // num_head
                  S, S' is sequence length
            when this layer used at source attention,
                 S is target's sequence length
                 S' is source's sequence length
        """
        # attn [B, S, num_head, num_head]
        attn = torch.matmul(q / self.temparature, k.transpose(-2, -1))
        if mask is not None:
            attn = attn.data.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim = -1))

        # output [B, S, num_head, d_k]
        output = torch.matmul(attn, v)
        return output, attn
