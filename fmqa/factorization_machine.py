"""
Factorization Machine implemented with PyTorch.
"""

from __future__ import annotations

__all__ = [
    "FactorizationMachineBinaryQuadraticModel", "FMBQM"
]

from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


def _resolve_backend(F=None):
    return torch if F is None else F


def _to_tensor(x, *, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def triu_mask(input_size, F=None):
    """Generate a square matrix with upper triangular elements set to 1 and others 0."""
    backend = _resolve_backend(F)
    if backend is np:
        return np.triu(np.ones((input_size, input_size), dtype=np.float32), k=1)
    mask = torch.ones((input_size, input_size), dtype=torch.float32)
    return torch.triu(mask, diagonal=1)


def VtoQ(V, F=None):
    """Calculate interaction strength by inner product of feature vectors."""
    backend = _resolve_backend(F)
    if backend is np:
        V_np = np.asarray(V, dtype=np.float32)
        Q = V_np.T @ V_np
        return Q * triu_mask(V_np.shape[1], np)

    V_tensor = _to_tensor(V)
    Q = V_tensor.transpose(0, 1) @ V_tensor
    return Q * triu_mask(V_tensor.shape[1], torch)


class QuadraticLayer(nn.Module):
    """A neural network layer which applies a quadratic function on the input.

    This class defines train() method for easy use.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def init_params(self, initializer: Optional[Callable[[torch.Tensor], None]] = None):
        """Initialize all parameters.

        Args:
            initializer:
                Callable applied to each parameter tensor. When omitted, a normal
                initialization close to the MXNet default is used.
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if initializer is not None:
                    initializer(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=1.0)
        self.optimizer = None

    def train(self, x=None, y=None, num_epoch=100, learning_rate=1.0e-2):
        """Training of the regression model using Adam optimizer.

        When called without x/y, behaves like torch.nn.Module.train(mode=True).
        """
        if x is None and y is None:
            return super().train(True)

        x_tensor = _to_tensor(x)
        y_tensor = _to_tensor(y).reshape(-1)
        batchsize = x_tensor.shape[0]
        super().train(True)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

        for _ in range(num_epoch):
            self.optimizer.zero_grad()
            output = self(x_tensor).reshape(-1)
            loss = torch.mean((y_tensor - output) ** 2)
            loss.backward()
            self.optimizer.step()

        return batchsize

    def get_bhQ(self):
        raise NotImplementedError()


class FactorizationMachine(QuadraticLayer):
    """Factorization Machine as a neural network layer.

    Args:
        input_size (int):
            The dimension of input value.
        factorization_size (int (<=input_size)):
            The rank of decomposition of interaction terms.
        act (string, optional):
            Name of activation function applied on FM output: "identity", "sigmoid", or "tanh".
    """

    def __init__(self, input_size, factorization_size=8, act="identity", **kwargs):
        super().__init__(**kwargs)
        self.factorization_size = factorization_size
        self.input_size = input_size
        self.h = nn.Parameter(torch.empty(input_size, dtype=torch.float32))
        if factorization_size > 0:
            self.V = nn.Parameter(torch.empty(factorization_size, input_size, dtype=torch.float32))
        else:
            self.V = nn.Parameter(torch.empty(1, input_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.act = act
        self.init_params()

    def forward(self, x):
        """Forward propagation of FM.

        Args:
          x: input vector of shape (N, d).
        """
        x_tensor = _to_tensor(x)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)

        linear_term = x_tensor @ self.h
        if self.factorization_size <= 0:
            return self.bias + linear_term

        Q = VtoQ(self.V)  # (d, d)
        Qx = x_tensor @ Q
        raw_output = self.bias + linear_term + torch.sum(x_tensor * Qx, dim=1)
        act = {
            "identity": lambda z: z,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }[self.act]
        return act(raw_output)

    def get_bhQ(self):
        """Returns linear and quadratic coefficients."""
        with torch.no_grad():
            if self.factorization_size == 0:
                V = torch.zeros_like(self.V)
            else:
                V = self.V.detach().clone()
            Q = VtoQ(V).cpu().numpy()
            return float(self.bias.detach().cpu().item()), self.h.detach().cpu().numpy().copy(), Q
