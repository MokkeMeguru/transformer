from nltk import bleu_score
from typing import List, TypeVar, Generic
from ..datasets.vocab import EOS

T = TypeVar('T')


class BLEUScore:
    def __init__(
            self,
            smoothing_function=bleu_score.SmoothingFunction().method7):
        self.smoothing_function = smoothing_function

    def calc_bleu(self, refs: List[List[T]], hyps: List[List[T]]):
        """
        Args:
            refs: reference sentences splitted by word / word_idx
            hyps: generated sentences splitted by word / word_idx
        Returns:
            bleu_score (float): [0, 100] score (upper is better)
        """
        refs = [[ref[:ref.index(EOS)]] for ref in refs]
        hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
        return 100 * bleu_score.corpus_bleu(
            refs, hyps, smoothing_function=self.smoothing_function)
