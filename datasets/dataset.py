import sys
sys.path.append("..")
sys.path.append(".")
from torchtext import data, datasets
from .vocab import Vocab, BOS, EOS, PAD
from typing import Dict
from pathlib import Path
from .custom_field import CustomField
import logging
from logging import getLogger
logger = getLogger(__name__)

def get_dataloader(train_data_base: str,
                   val_data_base: str,
                   test_data_base: str, ext: Dict):

    # load english
    en_vocab = Vocab()
    train_data_en = train_data_base + "." + ext["en"]
    with Path(train_data_en).open("r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f]
    en_vocab.build_vocab(sentences)

    # load japanese
    ja_vocab = Vocab()
    train_data_ja = train_data_base + "." + ext["ja"]
    with Path(train_data_ja).open("r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f]
    ja_vocab.build_vocab(sentences)

    src = CustomField(vocab=en_vocab,
                      bos_token=None, eos_token=None,
                      lower=True,
                      tokenize=lambda x: x.strip().split(),
                      batch_first=True)

    tgt = CustomField(vocab=ja_vocab,
                      lower=False,
                      tokenize=lambda x: x.strip().split(),
                      batch_first=True)
    train_dataloader = datasets.TranslationDataset(
        path=train_data_base, exts=("." + ext["en"], "."+ ext["ja"]),
        fields=(src, tgt))

    val_dataloader = datasets.TranslationDataset(
        path=val_data_base, exts=("." + ext["en"], "." + ext["ja"]),
        fields=(src, tgt)
    )

    test_dataloader = data.TabularDataset(
        path=test_data_base + "." + ext["en"], format="tsv",
        fields=[('text', src)]
    )

    return (train_dataloader, val_dataloader, test_dataloader), (en_vocab, ja_vocab)


def test():
    dataloaders, vocabs = get_dataloader("data/train", "data/dev", "data/test", ext={"ja":"ja", "en":"en"})
    print(list(vocabs[0].id2word.items())[:10])
    train_iter = data.BucketIterator(dataloaders[0], batch_size=10, sort_key= lambda x: data.interleave_keys(len(x.src), len(x.trg)))
    batch = next(iter(train_iter))
    print(batch)
    print(batch.src)
    print(batch.trg)

if __name__ == '__main__':
    test()
