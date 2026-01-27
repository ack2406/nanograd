from collections.abc import Callable

import numpy as np
from numpy.typing import DTypeLike, NDArray

float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)


# --- no_grad config ---


class Config:
    use_grad = True


class no_grad:
    def __enter__(self):
        self.prev = Config.use_grad
        Config.use_grad = False

    def __exit__(self, exc_type, exc_value, traceback):
        Config.use_grad = self.prev


# --- core ---


class Tensor:
    def __init__(
        self,
        data: list | NDArray | int | float,
        dtype: DTypeLike = float32,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
    ):
        self.data: NDArray = np.array(data, dtype=dtype)
        self.grad: NDArray | None = None
        self.requires_grad: bool = requires_grad
        self._prev: set[Tensor] = set(_children)
        self._backward: Callable[[], None] = lambda: None

    # --- helpers ---

    def _ensure_tensor(self, other: Tensor | NDArray | int | float | list) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)

    def _accumulate_grad(self, grad: NDArray | None) -> None:
        if self.requires_grad and grad is not None:
            self.grad = self.grad + grad if self.grad is not None else grad

    def _unbroadcast_expand(
        self, grad: NDArray | None, axis: int | None = None, keepdims: bool = False
    ) -> NDArray | None:
        if grad is None:
            return None
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis=axis)
        if grad.shape != self.data.shape:
            grad = np.broadcast_to(grad, self.data.shape)
        return grad

    def _unbroadcast_reduce(self, grad: NDArray | None) -> NDArray | None:
        if grad is None:
            return None
        ndims_added = grad.ndim - self.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        for i, (grad_dim, self_dim) in enumerate(zip(grad.shape, self.data.shape)):
            if self_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # --- autograd ---

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), dtype=self.data.dtype, requires_grad=False)

    # --- arithmetic ---

    def _get_requires_grad(self, other: Tensor) -> bool:
        return Config.use_grad and (self.requires_grad or other.requires_grad)

    def __add__(self, other: Tensor | int | float) -> Tensor:
        other = self._ensure_tensor(other)
        out = Tensor(self.data + other.data, dtype=self.data.dtype, _children=(self, other))
        out.requires_grad = self._get_requires_grad(other)

        def _backward() -> None:
            self._accumulate_grad(self._unbroadcast_reduce(out.grad))
            other._accumulate_grad(other._unbroadcast_reduce(out.grad))

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        other = self._ensure_tensor(other)
        out = Tensor(self.data * other.data, dtype=self.data.dtype, _children=(self, other))
        out.requires_grad = self._get_requires_grad(other)

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(self._unbroadcast_reduce(out.grad * other.data))
                other._accumulate_grad(other._unbroadcast_reduce(out.grad * self.data))

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor | NDArray) -> Tensor:
        other = self._ensure_tensor(other)
        out = Tensor(self.data @ other.data, dtype=self.data.dtype, _children=(self, other))
        out.requires_grad = self._get_requires_grad(other)

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad @ other.data.T)
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __pow__(self, power: int | float) -> Tensor:
        out = Tensor(self.data**power, dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad * power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        return self + (-other)

    def __truediv__(self, other: Tensor | int | float) -> Tensor:
        return self * (other**-1)

    def __radd__(self, other: int | float) -> Tensor:
        return self + other

    def __rmul__(self, other: int | float) -> Tensor:
        return self * other

    def __rsub__(self, other: int | float) -> Tensor:
        return (-self) + other

    def __rtruediv__(self, other: int | float) -> Tensor:
        return other * (self**-1)

    # --- reductions ---

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            self._accumulate_grad(self._unbroadcast_expand(out.grad, axis, keepdims))

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                grad = out.grad / (self.data.size if axis is None else self.data.shape[axis])
                self._accumulate_grad(self._unbroadcast_expand(grad, axis, keepdims))

        out._backward = _backward
        return out

    # --- math ---

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def sqrt(self) -> Tensor:
        return self**0.5

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(0, self.data), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad * out.data * (1 - out.data))

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(t, dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad * (1 - t**2))

        out._backward = _backward
        return out

    # --- shape ---

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(*shape), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        out = Tensor(np.transpose(self.data, axes), dtype=self.data.dtype, _children=(self,))
        out.requires_grad = Config.use_grad and self.requires_grad

        def _backward() -> None:
            if out.grad is not None:
                if axes is None:
                    self._accumulate_grad(out.grad.T)
                else:
                    self._accumulate_grad(np.transpose(out.grad, np.argsort(axes)))

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        return self.transpose()

    # --- utils ---

    def item(self) -> int | float:
        return self.data.item()

    def round(self, decimals: int = 0) -> Tensor:
        return Tensor(np.round(self.data, decimals), dtype=self.data.dtype, requires_grad=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        return f"Tensor({self.data})"


def tensor(data: list | NDArray | int | float, dtype: DTypeLike = float32, requires_grad: bool = False) -> Tensor:
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)
