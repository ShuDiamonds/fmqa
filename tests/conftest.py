import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dimod")


@pytest.fixture(autouse=True)
def fixed_random_seed():
    """各テストで乱数シードを固定し、学習結果のぶれを抑える。"""
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)
    yield
