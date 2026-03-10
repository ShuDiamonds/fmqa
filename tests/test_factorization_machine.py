import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from fmqa.factorization_machine import FactorizationMachine, VtoQ, triu_mask


def test_triu_mask_generates_expected_upper_triangle():
    """
    目的:
        上三角マスク生成が期待通りか確認する。

    確認内容:
        - 指定サイズの正方行列になること
        - 対角成分が0であること
        - 下三角が0であること
        - 上三角のみ1であること
        - input_size=4 のとき期待値と一致すること
    """
    actual = triu_mask(4, torch).detach().cpu().numpy()
    expected = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert actual.shape == (4, 4)
    np.testing.assert_allclose(actual, expected)
    assert np.all(np.diag(actual) == 0.0)
    assert np.all(np.tril(actual) == 0.0)
    assert np.all(actual[np.triu_indices(4, k=1)] == 1.0)


def test_vtoq_returns_upper_triangular_inner_product():
    """
    目的:
        特徴ベクトル行列 V から相互作用行列 Q が正しく生成されるか確認する。

    確認内容:
        - Q = V.T @ V の上三角成分のみが残ること
        - 対角および下三角が0になること
        - 小さな固定行列で期待値一致すること
    """
    V = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    actual = VtoQ(V, torch).detach().cpu().numpy()
    expected = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(actual, expected)
    assert np.all(np.diag(actual) == 0.0)
    assert np.all(np.tril(actual) == 0.0)


def test_factorization_machine_initialization_shapes():
    """
    目的:
        モデル生成時の内部パラメータ構造が壊れていないことを確認する。

    確認内容:
        - input_size が保持されること
        - factorization_size > 0 のとき V の shape が (k, d) になること
        - h, bias が存在すること
    """
    model = FactorizationMachine(input_size=5, factorization_size=3)
    model.init_params()

    assert model.input_size == 5
    assert model.factorization_size == 3
    assert tuple(model.h.shape) == (5,)
    assert tuple(model.V.shape) == (3, 5)
    assert tuple(model.bias.shape) == (1,)


def test_get_bhQ_returns_expected_shapes_and_upper_triangle():
    """
    目的:
        学習後または初期化後に、係数取得APIが期待フォーマットを返すことを保証する。

    確認内容:
        - bias が scalar であること
        - h が長さ input_size の1次元配列であること
        - Q が (input_size, input_size) の行列であること
        - Q が上三角のみ非ゼロになりうること
    """
    model = FactorizationMachine(input_size=4, factorization_size=2)
    model.init_params()

    bias, h, Q = model.get_bhQ()

    assert np.isscalar(bias)
    assert h.shape == (4,)
    assert Q.shape == (4, 4)
    assert np.all(np.diag(Q) == 0.0)
    assert np.all(np.tril(Q) == 0.0)


def test_factorization_machine_zero_factorization_returns_zero_q():
    """
    目的:
        factorization_size=0 のとき二次項なしの線形モデルとして動くことを確認する。

    確認内容:
        - get_bhQ() の Q がゼロ行列になること
        - forward が線形項 + bias ベースで動くこと
        - 学習後もエラーなく予測できること
    """
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0.2, 1.2, -0.8, 0.2], dtype=np.float32)

    model = FactorizationMachine(input_size=2, factorization_size=0)
    model.init_params()
    model.train(x, y, num_epoch=150, learning_rate=0.05)

    bias, h, Q = model.get_bhQ()
    pred = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()

    assert np.isscalar(bias)
    assert h.shape == (2,)
    np.testing.assert_allclose(Q, np.zeros((2, 2), dtype=np.float32))
    assert pred.shape == (4,)


def test_factorization_machine_training_reduces_mse_for_linear_data():
    """
    目的:
        train() が最低限機能し、学習前より学習後の誤差が改善することを確認する。

    確認内容:
        - 小さな人工データに対して学習前より学習後の MSE が下がること
        - 完全一致ではなく改善傾向を確認すること
    """
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )
    y = 0.5 + 1.5 * x[:, 0] - 0.75 * x[:, 1]

    model = FactorizationMachine(input_size=2, factorization_size=0, act="identity")
    model.init_params()

    before = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    before_mse = np.mean((y - before) ** 2)

    model.train(x, y, num_epoch=250, learning_rate=0.05)

    after = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    after_mse = np.mean((y - after) ** 2)

    assert after_mse < before_mse


@pytest.mark.parametrize("act", ["identity", "sigmoid", "tanh"])
def test_factorization_machine_supports_activation_options(act):
    """
    目的:
        act 引数でモデル生成・学習・予測が可能であることを確認する。

    確認内容:
        - act="identity", "sigmoid", "tanh" の各設定で初期化できること
        - 学習および予測がエラーなく実行できること
    """
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y = np.array([0.0, 0.3, -0.2], dtype=np.float32)

    model = FactorizationMachine(input_size=2, factorization_size=1, act=act)
    model.init_params()
    model.train(x, y, num_epoch=5, learning_rate=0.01)
    pred = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()

    assert pred.shape == (3,)


def test_factorization_machine_invalid_activation_raises_keyerror():
    """
    目的:
        不正な act が指定された場合の現仕様を固定する。

    確認内容:
        - 未対応の act を指定して forward を呼ぶと KeyError になること
    """
    model = FactorizationMachine(input_size=2, factorization_size=1, act="relu")
    model.init_params()

    with pytest.raises(KeyError):
        model(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
