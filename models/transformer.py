import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.embedding import Embedding
from .layers.encoder import Encoder
from .layers.decoder import Decoder
from .layers.posencoding import PositionalEncoding
from typing import Dict

class Transformer(nn.Module):
    def __init__(self,
                 basic_params: Dict,
                 encoder_params: Dict,
                 decoder_params: Dict,
                 src_pad_idx: int=0,
                 tgt_pad_idx: int=0):
        super().__init__()
        self.src_embed = Embedding(
            basic_params["transformer"]["d_model"],
            basic_params["vocab"]["num_src_vocab"],
            pad_idx=src_pad_idx)
        self.tgt_embed = Embedding(
            basic_params["transformer"]["d_model"],
            basic_params["vocab"]["num_tgt_vocab"],
            pad_idx=tgt_pad_idx)

        self.pos_encoding = PositionalEncoding(
            basic_params["transformer"]["d_model"],
            dropout_rate=basic_params["transformer"]["posencoding"]["dropout_rate"])

        self.generator = nn.Linear(
            basic_params["transformer"]["d_model"],
            basic_params["vocab"]["num_src_vocab"],
            bias=False)

        if basic_params["transformer"]["proj_share_weight"]:
            self.generator.weight = self.tgt_embed.lut.weight

        self.encoder = Encoder(
            max_length=basic_params["src_max_seq_len"],
            num_layer=encoder_params["num_layer"],
            num_head=encoder_params["num_head"],
            d_model=basic_params["transformer"]["d_model"],
            d_ff=encoder_params["d_ff"],
            dropout_rate=encoder_params["dropout_rate"])

        self.decoder = Decoder(
            max_length=basic_params["tgt_max_seq_len"],
            num_layer=decoder_params["num_layer"],
            num_head=decoder_params["num_head"],
            d_model=basic_params["transformer"]["d_model"],
            d_ff=decoder_params["d_ff"],
            dropout_rate=decoder_params["dropout_rate"])

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor):

        src = self.pos_encoding(self.src_embed(src))
        tgt = self.pos_encoding(self.tgt_embed(tgt))

        enc_output, *_ = self.encoder(src, src_mask)
        dec_output, *_ = self.decoder(
            tgt, enc_output, tgt_mask, src_mask)
        output = F.log_softmax(self.generator(dec_output), dim=-1)
        return output
