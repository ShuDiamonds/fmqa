"""Annealing backends for FMBQM.

Supports:
- dimod simulated annealing
- TYTAN simulated annealing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fm_binary_quadratic_model import FMBQM


@dataclass
class AnnealResult:
    """Common annealing result container."""

    samples: np.ndarray
    energies: np.ndarray
    counts: np.ndarray
    backend: str
    raw: Any


class Annealer:
    """Backend selector for annealing FMBQM models."""

    @staticmethod
    def sample(model: FMBQM, backend: str = "dimod-sa", num_reads: int = 1, seed: int | None = None, **kwargs) -> AnnealResult:
        backend_key = backend.lower()
        if backend_key in {"dimod", "dimod-sa", "dwave-sa", "d-wave-sa", "sa"}:
            return Annealer._sample_with_dimod(model, num_reads=num_reads, seed=seed, **kwargs)
        if backend_key in {"tytan", "tytan-sa"}:
            return Annealer._sample_with_tytan(model, num_reads=num_reads, seed=seed, **kwargs)
        raise ValueError(f"Unsupported annealing backend: {backend}")

    @staticmethod
    def _sample_with_dimod(model: FMBQM, num_reads: int = 1, seed: int | None = None, **kwargs) -> AnnealResult:
        import dimod

        sampler = dimod.samplers.SimulatedAnnealingSampler()
        sample_kwargs = dict(kwargs)
        if seed is not None:
            sample_kwargs.setdefault("seed", seed)
        res = sampler.sample(model, num_reads=num_reads, **sample_kwargs)
        samples = np.asarray(res.record["sample"], dtype=np.int64)
        energies = np.asarray(res.record["energy"], dtype=float)
        counts = np.asarray(res.record["num_occurrences"], dtype=np.int64)
        return AnnealResult(samples=samples, energies=energies, counts=counts, backend="dimod-sa", raw=res)

    @staticmethod
    def _sample_with_tytan(model: FMBQM, num_reads: int = 1, seed: int | None = None, **kwargs) -> AnnealResult:
        try:
            from tytan import sampler as tytan_sampler
        except ImportError as exc:
            raise ImportError(
                "TYTAN is not installed. Install it with 'pip install tytan' or "
                "'pip install git+https://github.com/tytansdk/tytan'."
            ) from exc

        qubo, offset = model.to_qubo()
        qmatrix, index_map = Annealer._qubo_dict_to_tytan_input(qubo, model.fm.input_size)

        sampler = tytan_sampler.SASampler(seed=seed)
        res = sampler.run(
            (qmatrix, index_map),
            shots=num_reads,
            T_num=kwargs.get("T_num", 2000),
            show=kwargs.get("show", False),
        )

        samples = []
        energies = []
        counts = []
        ordered_keys = list(index_map.keys())
        for entry in res:
            sample_dict = entry[0]
            energy = float(entry[1]) + float(offset)
            count = int(entry[2])
            sample = [int(sample_dict[key]) for key in ordered_keys]
            samples.append(sample)
            energies.append(energy)
            counts.append(count)

        return AnnealResult(
            samples=np.asarray(samples, dtype=np.int64),
            energies=np.asarray(energies, dtype=float),
            counts=np.asarray(counts, dtype=np.int64),
            backend="tytan-sa",
            raw=res,
        )

    @staticmethod
    def _qubo_dict_to_tytan_input(qubo: dict[tuple[int, int], float], size: int) -> tuple[np.ndarray, dict[str, int]]:
        qmatrix = np.zeros((size, size), dtype=float)
        for (i, j), value in qubo.items():
            qmatrix[i, j] = float(value)

        index_map = {f"x{i}": i for i in range(size)}
        return qmatrix, index_map


__all__ = ["Annealer", "AnnealResult"]
