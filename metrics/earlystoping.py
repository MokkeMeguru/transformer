import numpy as np
from logging import getLogger
from typing import Callable
logger = getLogger(__name__)


class EarlyStopping:
    """Early stops the training
    if validation loss does'nt improve after a given patience.

    Examples:
        >>> early_stopping = EarlyStopping()
        >>> model.save() if early_stopping(val_loss=val_loss)
        >>> stop_training() if early_stopping.early_stop
    """
    def __init__(self, patience: int=7, delta:float=0.0):
        """
        Args:
            patience: how long to waint
                      after last time validation loss improved
            delta   : minimum change in the monitored quantity
                      to qualify as an improvement
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss: float):
        """
        Args:
            val_loss: validation loss (lower is better)
        Returns:
            best_score? (bool): if it's the best score, return True
        """
        score = - val_loss
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True
