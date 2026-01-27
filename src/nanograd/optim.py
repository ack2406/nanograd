import numpy as np
from numpy.typing import NDArray

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


# --- optimizers ---


class SGD(Optimizer):
    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad


class AdamW(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state: dict[Tensor, dict[str, NDArray]] = {}

    def step(self) -> None:
        self.t += 1
        beta1, beta2 = self.betas

        for p in self.params:
            if p.grad is None:
                continue

            if p not in self.state:
                self.state[p] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }

            m = self.state[p]["m"]
            v = self.state[p]["v"]
            grad = p.grad

            p.data = p.data - self.lr * self.weight_decay * p.data

            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad**2)

            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)

            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
