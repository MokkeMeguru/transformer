import torch


class NormOpt:
    def __init__(self, d_model: int, factor: float, warmup_step: int,
                 optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_step = warmup_step
        self.d_model = d_model
        self._rate = 0
        self.factor = factor

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.d_model ** (-0.5) *
             min(step ** (-0.5), step * self.warmup_step ** (-1.5)))
    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(model,
                optimizer: torch.optim.Optimizer,
                factor: float = 2.0,
                warmup_step: int = 4000):
    return NormOpt(model.src_embed[0].d_model, factor, warmup_step, optimizer)
