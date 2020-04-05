import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scaled_dot_prod_attention import ScaledDotProdAttention
import logging
from logging import getLogger
logger = getLogger(__name__)


class TestSDPA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSDPA, self).__init__(*args, **kwargs)
        self.batch_size = 32
        self.d_model = 12
        self.num_head = 3
        self.src_len = 16
        self.trg_len = 10
        self.sdpa = ScaledDotProdAttention(self.d_model, dropout_rate=0.1)

    def test_self_attn(self):
        query = torch.randn(
            self.batch_size, self.src_len, self.num_head,
            self.d_model // self.num_head).transpose(1, 2)
        key = torch.randn(
            self.batch_size, self.src_len, self.num_head,
            self.d_model // self.num_head).transpose(1, 2)
        value = torch.randn(
            self.batch_size, self.src_len, self.num_head,
            self.d_model // self.num_head).transpose(1, 2)
        out, attn = self.sdpa(query, key, value, mask = None)
        pred_out_shape = (self.batch_size, self.num_head,
                          self.src_len, self.d_model // self.num_head)
        pred_attn_shape = (self.batch_size, self.num_head,
                           self.src_len, self.src_len)
        assert tuple(out.shape) == pred_out_shape,\
            "{} vs {} (correct)".format(out.shape, pred_out_shape)
        assert tuple(attn.shape) == pred_attn_shape,\
            "{} vs {} (correct)".format(attn.shape, pred_attn_shape)

    def test_src_attn(self):
        query = torch.randn(
            self.batch_size, self.trg_len, self.num_head,
            self.d_model // self.num_head).transpose(1, 2)
        key = torch.randn(
            self.batch_size, self.src_len, self.num_head,
            self.d_model // self.num_head).transpose(1, 2)
        value = torch.randn(
            self.batch_size, self.src_len, self.num_head,
            self.d_model // self.num_head).transpose(1, 2)
        out, attn = self.sdpa(query, key, value, mask = None)
        pred_out_shape = (self.batch_size, self.num_head,
                          self.trg_len, self.d_model // self.num_head)
        pred_attn_shape = (self.batch_size, self.num_head,
                           self.trg_len, self.src_len)
        assert tuple(out.shape) == pred_out_shape,\
            "{} vs {} (correct)".format(out.shape, pred_out_shape)
        assert tuple(attn.shape) == pred_attn_shape,\
            "{} vs {} (correct)".format(attn.shape, pred_attn_shape)


if __name__ == '__main__':
    unittest.main()
