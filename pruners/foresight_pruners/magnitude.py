# coding=utf-8

from tools import named_params

from .pruner import Pruner

__all__ = [
    "Magnitude",
]


class Magnitude(Pruner):
    """
    Magnitude model pruner
    """

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        params = named_params(self._model)
        for name, score in self.scores.items():
            param = params[name]
            score.copy_(param.detach().abs())
