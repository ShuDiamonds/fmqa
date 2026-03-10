import sys
import types

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("dimod")

import fmqa
from fmqa.annealing import Annealer


@pytest.fixture
def trained_model():
    x = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0.0, 1.0, -1.0, 0.5, 0.2, 2.0, -0.1, 1.2], dtype=np.float32)
    return fmqa.FMBQM.from_data(x, y, num_epoch=40, learning_rate=0.03)


def test_dimod_backend_returns_common_result(trained_model):
    res = Annealer.sample(trained_model, backend="dimod-sa", num_reads=4, seed=0)

    assert res.backend == "dimod-sa"
    assert isinstance(res.samples, np.ndarray)
    assert isinstance(res.energies, np.ndarray)
    assert isinstance(res.counts, np.ndarray)
    assert res.samples.shape[1] == trained_model.fm.input_size
    assert len(res.samples) == len(res.energies) == len(res.counts)


def test_tytan_backend_uses_optional_sampler(monkeypatch, trained_model):
    class DummySASampler:
        def __init__(self, seed=None):
            self.seed = seed

        def run(self, qubomix, shots=100, T_num=2000, show=False):
            _, index_map = qubomix
            keys = list(index_map.keys())
            return [
                [{key: 0 for key in keys}, -1.0, shots],
                [{key: 1 for key in keys}, 0.5, 1],
            ]

    fake_tytan = types.ModuleType("tytan")
    fake_tytan.sampler = types.SimpleNamespace(SASampler=DummySASampler)
    monkeypatch.setitem(sys.modules, "tytan", fake_tytan)

    res = Annealer.sample(trained_model, backend="tytan-sa", num_reads=3, seed=123)

    assert res.backend == "tytan-sa"
    assert res.samples.shape == (2, trained_model.fm.input_size)
    assert res.counts.tolist() == [3, 1]
    assert set(np.unique(res.samples)).issubset({0, 1})


def test_tytan_backend_requires_optional_dependency(monkeypatch, trained_model):
    monkeypatch.delitem(sys.modules, "tytan", raising=False)

    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "tytan":
            raise ImportError("missing tytan")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    with pytest.raises(ImportError):
        Annealer.sample(trained_model, backend="tytan-sa", num_reads=2)
