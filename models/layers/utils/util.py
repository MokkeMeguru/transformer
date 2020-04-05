"""some utilities
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x: torch.Tensor, volatile: bool = False):
    """torch tensor to torch.autograd.Variable
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def square_subsequent_mask(size):
    """subsequent mask
    ref. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    If mask is invalid, PyTorch is invalid
    """
    # mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    # mask = mask.bool().masked_fill(mask == 0, False)\
    #                   .masked_fill(mask == 1, True)
    mask = (1 - torch.triu(
        torch.ones((1, size, size)), diagonal=1)).bool()
    return mask


def padding_mask(src: torch.Tensor, pad_id: int = 0):
    """
    Args:
       src: source [B, S]
    Examples:
       >>> padding_mask(torch.Tensor([[1, 2, 3, 0], [0, 0, 0, 0]]))
       tensor([[[ True,  True,  True, False]],
               [[False, False, False, False]]])
    """
    src_mask = (src != pad_id).unsqueeze(-2)
    return src_mask

def look_ahead_mask(tgt: torch.Tensor, pad_id: int = 0):
    """
    Args:
       tgt: target [B, S]

    Examples:
        >>> look_ahead_mask(torch.Tensor([[1, 2, 3, 0], [0, 0, 0, 0]]))
        tensor([[[ True, False, False, False],
                 [ True,  True, False, False],
                 [ True,  True,  True, False],
                 [ True,  True,  True, False]],

                [[False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False]]])
    """
    tgt_mask = (tgt != pad_id).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        square_subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
