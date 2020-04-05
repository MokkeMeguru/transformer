from datasets import dataset
import logging
from logging import getLogger
logger = getLogger(__name__)

def dataset_test():
    dataloaders, vocabs = dataset.get_dataloader("data/train", "data/dev", "data/test", ext={"ja":"ja", "en":"en"})
    dataset.test()


if __name__ == '__main__':
    dataset_test()
