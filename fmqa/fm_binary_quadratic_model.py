"""
Trainable Binary Quadratic Model based on Factorization Machine (FMBQM)
"""

from __future__ import annotations

import numpy as np
import torch
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.vartypes import Vartype

from .factorization_machine import FactorizationMachine

__all__ = [
    "FactorizationMachineBinaryQuadraticModel",
    "FMBQM",
]


class FactorizationMachineBinaryQuadraticModel(BinaryQuadraticModel):
    """FMBQM: Trainable BQM based on Factorization Machine."""

    def __init__(self, input_size, vartype, act="identity", **kwargs):
        fm_kwargs = dict(kwargs)
        init_linear = {i: 0.0 for i in range(input_size)}
        init_quadratic = {}
        init_offset = 0.0
        super().__init__(init_linear, init_quadratic, init_offset, vartype)
        self.fm = FactorizationMachine(input_size, act=act, **fm_kwargs)

    def to_qubo(self):
        return self._fm_to_qubo()

    def to_ising(self):
        return self._fm_to_ising()

    @classmethod
    def from_data(cls, x, y, act="identity", num_epoch=1000, learning_rate=1.0e-2, **kwargs):
        """Create a binary quadratic model by FM regression model trained on the provided data."""
        x_np = np.asarray(x)
        y_np = np.asarray(y, dtype=np.float32)

        if np.all((x_np == 0) | (x_np == 1)):
            vartype = Vartype.BINARY
        elif np.all((x_np == -1) | (x_np == 1)):
            vartype = Vartype.SPIN
        else:
            raise ValueError("input data should BINARY or SPIN vectors")

        input_size = x_np.shape[-1]
        fmbqm = cls(input_size, vartype, act, **kwargs)
        fmbqm.train(x_np, y_np, num_epoch, learning_rate, init=True)
        return fmbqm

    def train(self, x, y, num_epoch=1000, learning_rate=1.0e-2, init=False):
        """Train FM regression model on the provided data."""
        x_np = np.asarray(x)
        y_np = np.asarray(y, dtype=np.float32)
        if init:
            self.fm.init_params()
        self._check_vartype(x_np)
        self.fm.train(x_np, y_np, num_epoch, learning_rate)
        if self.vartype == Vartype.SPIN:
            h, J, b = self._fm_to_ising()
            self.offset = float(b)
            for i in range(self.fm.input_size):
                self.linear[i] = float(h[i])
                for j in range(i + 1, self.fm.input_size):
                    self.quadratic[(i, j)] = float(J.get((i, j), 0.0))
        elif self.vartype == Vartype.BINARY:
            Q, b = self._fm_to_qubo()
            self.offset = float(b)
            for i in range(self.fm.input_size):
                self.linear[i] = float(Q[(i, i)])
                for j in range(i + 1, self.fm.input_size):
                    self.quadratic[(i, j)] = float(Q.get((i, j), 0.0))

    def predict(self, x):
        """Predict target value by trained model."""
        x_np = np.asarray(x)
        self._check_vartype(x_np)
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
        with torch.no_grad():
            return self.fm(x_tensor).cpu().numpy()

    def _check_vartype(self, x):
        x_np = np.asarray(x)
        if (
            (self.vartype is Vartype.BINARY and np.all((1 == x_np) | (0 == x_np)))
            or (self.vartype is Vartype.SPIN and np.all((1 == x_np) | (-1 == x_np)))
        ):
            return
        raise ValueError("input data should be of type", self.vartype)

    def _fm_to_ising(self, scaling=True):
        """Convert trained model into Ising parameters."""
        b, h, J = self.fm.get_bhQ()
        if self.vartype is Vartype.BINARY:
            b = b + np.sum(h) / 2 + np.sum(J) / 4
            h = (2 * h + np.sum(J, axis=0) + np.sum(J, axis=1)) / 4.0
            J = J / 4.0
        if scaling:
            max_h = np.max(np.abs(h)) if h.size else 0.0
            max_J = np.max(np.abs(J)) if J.size else 0.0
            scaling_factor = max(max_h, max_J, 1.0)
            b /= scaling_factor
            h /= scaling_factor
            J /= scaling_factor
        return {key: h[key] for key in range(len(h))}, {key: J[key] for key in zip(*J.nonzero())}, float(b)

    def _fm_to_qubo(self, scaling=True):
        """Convert trained model into QUBO parameters."""
        b, h, Q = self.fm.get_bhQ()
        Q = np.array(Q, dtype=np.float32, copy=True)
        h = np.array(h, dtype=np.float32, copy=True)
        if self.vartype is Vartype.SPIN:
            b = b - np.sum(h) + np.sum(Q)
            h = 2 * (h - np.sum(Q, axis=0) - np.sum(Q, axis=1))
            Q = 4 * Q
        Q[np.diag_indices(len(Q))] = h
        if scaling:
            scaling_factor = max(np.max(np.abs(Q)), 1.0)
            b /= scaling_factor
            Q /= scaling_factor
        Q_dict = {key: Q[key] for key in zip(*Q.nonzero())}
        for i in range(len(Q)):
            Q_dict[(i, i)] = Q[i, i]
        return Q_dict, float(b)


FMBQM = FactorizationMachineBinaryQuadraticModel
