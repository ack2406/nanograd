import numpy as np

from .tensor import Tensor, float32


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> list[Tensor]:
        return []

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


# --- containers ---


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# --- layers ---


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * scale, dtype=float32, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), dtype=float32, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


# --- loss ---


class BCELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-12
        term1 = target * (pred + eps).log()
        term2 = (1 - target) * (1 - pred + eps).log()
        return -(term1 + term2).mean()
