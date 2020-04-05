import torch
from torchtext import data
from typing import Callable, List

from .vocab import (Vocab,
                    UNK, BOS, EOS, PAD, MASK,
                    UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)

class CustomField(data.RawField):
    def __init__(self,
                 vocab: Vocab = None,
                 bos_token = BOS_TOKEN,
                 eos_token = EOS_TOKEN,
                 preprocessing: Callable[[List[str]], List[str]]=None,
                 postprocessing: Callable[[List[int], Vocab], List[int]]=None,
                 dtype=torch.long,
                 lower: bool = False,
                 tokenize: Callable[[str], List[str]] = lambda x: x.split(),
                 truncate_first: bool = False,
                 pad_first: bool = False,
                 is_target: bool = False,
                 batch_first: bool= True,
                 fix_length: int = None):
        """
        Notes:
             fix_length is not included bos / eos if bos and eos are not None
             so fix_length should add 2 for bos / eos token
        """
        if vocab is None:
            raise NotImplementedError()
        self.vocab = vocab
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.dtype = dtype
        self.lower = lower
        self.tokenize = tokenize
        self.pad_first = pad_first
        self.batch_first = batch_first
        self.truncate_first = truncate_first
        self.fix_length = fix_length
        self.is_target = is_target

    def preprocess(self, sentence: str):
        sentence = self.tokenize(sentence)
        if self.preprocessing is not None:
            sentence = self.preprocessing(sentence)
        return sentence

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = (max(len(x) for x in minibatch)
                       + 2 -  (self.bos_token, self.eos_token).count(None))
        else:
            max_len = self.fix_length - 2 + (self.bos_token, self.eos_token).count(None)
        padded = []
        for sentence in minibatch:
            if self.pad_first:
                ids = [PAD_TOKEN for _ in range(max(0, max_len - len(ids)))]
                ids += [] if self.bos_token is None else [self.bos_token]
                ids += list(sentence[-max_len:]
                            if self.truncate_first else sentence[:max_len])
                ids += [] if self.eos_token is None else [self.eos_token]
            else:
                ids = [] if self.bos_token is None else [self.bos_token]
                ids += list(sentence[-max_len:]
                            if self.truncate_first else sentence[:max_len])
                ids += [] if self.eos_token is None else [self.eos_token]
                ids += [PAD_TOKEN for _ in range(max(0, max_len - len(ids)))]
            padded.append(ids)
        return padded

    def numericalize(self, arr: List[List[str]], device=None):
        arr = [self.vocab.encode(sent) for sent in arr]
        if self.postprocessing is not None:
            arr = self.postprocessing(arr, self, vocab)
        var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()
        return var
