from typing import Dict, List

PAD = 0
BOS = 1
EOS = 2
UNK = 3
MASK = 4

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"
MASK_TOKEN = "<MASK>"

default_word2id = {
    PAD_TOKEN: PAD,
    BOS_TOKEN: BOS,
    EOS_TOKEN: EOS,
    UNK_TOKEN: UNK,
    MASK_TOKEN: MASK
}

class Vocab:
    """
    Attributes:
        word2id: word    (str) -> word_idx(int)
        id2word: word_idx(int) -> word    (str)
    """
    def __init__(self, word2id: Dict=None):
        """
        Args:
            word2id: word (str) -> word_index(int)
        Note:
            word2id should includes default_word2id
        """
        if word2id is not None:
            self.word2id = dict(word2id)
        else:
            import copy
            self.word2id = copy.deepcopy(default_word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences: List[List[str]], min_freq=2):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1
        for word, count in sorted(
                word_counter.items(), key=lambda x: x[1], reverse=True):
            if count < min_freq:
                break
            idx = len(self.word2id)
            self.word2id.setdefault(word, idx)
            self.id2word[idx] = word

    def encode(self, sentence: List[str], add_bos_eos: bool=False):
        """
        """
        ids = [self.word2id.get(word, UNK) for word in sentence]
        if add_bos_eos:
            ids = [BOS] + ids + [EOS]
        return ids

    def decode(self, ids: List[int], remove_eos: bool = True):
        """
        """
        sentence = [self.id2word[idx] for idx in ids]
        if remove_eos and EOS_TOKEN in sentence:
            sentence = sentence[:sentence.index(EOS_TOKEN)]
        return sentence
