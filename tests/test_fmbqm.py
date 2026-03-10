import numpy as np
import pytest

pytest.importorskip("mxnet")
dimod = pytest.importorskip("dimod")

from dimod.vartypes import Vartype

from fmqa.fm_binary_quadratic_model import FMBQM


@pytest.fixture
def binary_dataset():
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
    return x, y


@pytest.fixture
def spin_dataset():
    x = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([-0.4, 0.9, -0.8, 0.2, 0.5, 1.1, -0.2, 0.7], dtype=np.float32)
    return x, y


def test_from_data_detects_binary_vartype(binary_dataset):
    """
    目的:
        {0,1} 入力を Vartype.BINARY と認識することを確認する。

    確認内容:
        - BINARYデータを渡したとき model.vartype == Vartype.BINARY になること
        - input_size が入力次元と一致すること
        - 学習後に線形項へアクセスできること
    """
    x, y = binary_dataset
    model = FMBQM.from_data(x, y, num_epoch=50, learning_rate=0.03)

    assert model.vartype == Vartype.BINARY
    assert model.fm.input_size == x.shape[1]
    assert len(model.linear) == x.shape[1]


def test_from_data_detects_spin_vartype(spin_dataset):
    """
    目的:
        {-1,1} 入力を Vartype.SPIN と認識することを確認する。

    確認内容:
        - SPINデータを渡したとき model.vartype == Vartype.SPIN になること
        - input_size が入力次元と一致すること
        - 学習後に線形項へアクセスできること
    """
    x, y = spin_dataset
    model = FMBQM.from_data(x, y, num_epoch=50, learning_rate=0.03)

    assert model.vartype == Vartype.SPIN
    assert model.fm.input_size == x.shape[1]
    assert len(model.linear) == x.shape[1]


def test_from_data_rejects_invalid_vartype_data():
    """
    目的:
        BINARY / SPIN 以外の入力が与えられたとき例外を出すことを確認する。

    確認内容:
        - 0,1,2 や小数を含む入力で ValueError になること
        - BINARY/SPIN混在の入力でも ValueError になること
    """
    invalid_cases = [
        np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int32),
        np.array([[0.2, 1.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[0, 1, -1], [1, 0, 1]], dtype=np.int32),
    ]
    y = np.array([0.0, 1.0], dtype=np.float32)

    for x in invalid_cases:
        with pytest.raises(ValueError):
            FMBQM.from_data(x, y, num_epoch=5)


def test_train_updates_bqm_coefficients_binary(binary_dataset):
    """
    目的:
        train() 実行後に BQM の内部係数が更新されることを確認する。

    確認内容:
        - 学習前後で線形項・二次項・offset の少なくとも一部が変化すること
        - 変数数分の線形項があること
        - 二次項辞書へアクセスできること
    """
    x, y = binary_dataset
    model = FMBQM(input_size=x.shape[1], vartype=Vartype.BINARY)

    before_linear = dict(model.linear)
    before_quadratic = dict(model.quadratic)
    before_offset = model.offset

    model.train(x, y, num_epoch=60, learning_rate=0.03, init=True)

    after_linear = dict(model.linear)
    after_quadratic = dict(model.quadratic)
    after_offset = model.offset

    assert len(after_linear) == x.shape[1]
    assert isinstance(after_quadratic, dict)
    assert (
        before_linear != after_linear
        or before_quadratic != after_quadratic
        or before_offset != after_offset
    )


def test_predict_returns_expected_shapes_for_batch_and_single(binary_dataset):
    """
    目的:
        predict() の返却 shape と型が期待通りであることを確認する。

    確認内容:
        - 2次元入力で (N,) 相当の予測配列が返ること
        - 1次元入力でも1サンプルとして扱われること
        - numpy.ndarray が返ること
    """
    x, y = binary_dataset
    model = FMBQM.from_data(x, y, num_epoch=40, learning_rate=0.03)

    batch_pred = model.predict(x)
    single_pred = model.predict(x[0])

    assert isinstance(batch_pred, np.ndarray)
    assert isinstance(single_pred, np.ndarray)
    assert batch_pred.shape == (len(x),)
    assert single_pred.shape == (1,)


def test_predict_rejects_mismatched_vartype(binary_dataset):
    """
    目的:
        モデルの vartype と異なる入力を predict() に渡したとき例外を出すことを確認する。

    確認内容:
        - BINARYモデルへ SPIN入力を与えると ValueError になること
    """
    x, y = binary_dataset
    model = FMBQM.from_data(x, y, num_epoch=20, learning_rate=0.03)
    invalid_spin_like = np.array([1, -1, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        model.predict(invalid_spin_like)


def test_check_vartype_accepts_valid_and_rejects_invalid(binary_dataset, spin_dataset):
    """
    目的:
        _check_vartype() の内部バリデーション仕様を固定する。

    確認内容:
        - BINARY / SPIN の正常ケースでは例外が出ないこと
        - 異常ケースでは ValueError になること
    """
    binary_x, _ = binary_dataset
    spin_x, _ = spin_dataset

    binary_model = FMBQM(input_size=3, vartype=Vartype.BINARY)
    spin_model = FMBQM(input_size=3, vartype=Vartype.SPIN)

    binary_model._check_vartype(binary_x)
    spin_model._check_vartype(spin_x)

    with pytest.raises(ValueError):
        binary_model._check_vartype(spin_x)
    with pytest.raises(ValueError):
        spin_model._check_vartype(binary_x)


def test_to_qubo_returns_expected_structure(binary_dataset):
    """
    目的:
        to_qubo() の返却形式を保証する。

    確認内容:
        - 戻り値が (Q_dict, offset) の2要素であること
        - Q_dict のキーが (i, j) タプルであること
        - 対角 (i, i) が含まれること
        - 値が数値型であること
    """
    x, y = binary_dataset
    model = FMBQM.from_data(x, y, num_epoch=40, learning_rate=0.03)

    Q_dict, offset = model.to_qubo()

    assert isinstance(Q_dict, dict)
    assert isinstance(offset, (int, float, np.floating))
    for i in range(x.shape[1]):
        assert (i, i) in Q_dict
    for key, value in Q_dict.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert all(isinstance(index, (int, np.integer)) for index in key)
        assert isinstance(value, (int, float, np.floating))


def test_to_ising_returns_expected_structure(spin_dataset):
    """
    目的:
        to_ising() の返却形式を保証する。

    確認内容:
        - 戻り値が (h_dict, J_dict, offset) の3要素であること
        - h_dict は各変数の線形項を持つこと
        - J_dict はペア項辞書であること
        - 値が数値型であること
    """
    x, y = spin_dataset
    model = FMBQM.from_data(x, y, num_epoch=40, learning_rate=0.03)

    h_dict, J_dict, offset = model.to_ising()

    assert isinstance(h_dict, dict)
    assert isinstance(J_dict, dict)
    assert isinstance(offset, (int, float, np.floating))
    assert set(h_dict.keys()) == set(range(x.shape[1]))
    for value in h_dict.values():
        assert isinstance(value, (int, float, np.floating))
    for key, value in J_dict.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(value, (int, float, np.floating))


def test_binary_and_spin_models_can_cross_convert(binary_dataset, spin_dataset):
    """
    目的:
        BINARY→Ising変換と SPIN→QUBO変換がどちらも実行可能であることを確認する。

    確認内容:
        - BINARYモデルで to_ising() が呼べること
        - SPINモデルで to_qubo() が呼べること
        - 返却値の形式が崩れないこと
    """
    binary_x, binary_y = binary_dataset
    spin_x, spin_y = spin_dataset

    binary_model = FMBQM.from_data(binary_x, binary_y, num_epoch=30, learning_rate=0.03)
    spin_model = FMBQM.from_data(spin_x, spin_y, num_epoch=30, learning_rate=0.03)

    h_dict, J_dict, b1 = binary_model.to_ising()
    Q_dict, b2 = spin_model.to_qubo()

    assert isinstance(h_dict, dict)
    assert isinstance(J_dict, dict)
    assert isinstance(b1, (int, float, np.floating))
    assert isinstance(Q_dict, dict)
    assert isinstance(b2, (int, float, np.floating))


def test_trained_model_can_be_sampled_by_exact_solver(binary_dataset):
    """
    目的:
        学習済み FMBQM が dimod サンプラーへ渡せることを確認する。

    確認内容:
        - 学習済みモデルを dimod.ExactSolver() に渡して sample() を実行できること
        - 返ってきたサンプルに全変数が含まれること
    """
    x, y = binary_dataset
    model = FMBQM.from_data(x, y, num_epoch=30, learning_rate=0.03)

    sampleset = dimod.ExactSolver().sample(model)

    assert len(sampleset) > 0
    assert set(sampleset.variables) == set(range(x.shape[1]))


def test_from_data_with_zero_factorization_still_predicts(binary_dataset):
    """
    目的:
        factorization_size=0 の特殊ケースでも FMBQM が学習・予測できることを確認する。

    確認内容:
        - factorization_size=0 で from_data() が成功すること
        - predict() がエラーなく実行できること
        - to_qubo() の対角項へアクセスできること
    """
    x, y = binary_dataset
    model = FMBQM.from_data(x, y, num_epoch=40, learning_rate=0.03, factorization_size=0)

    pred = model.predict(x)
    Q_dict, _ = model.to_qubo()

    assert pred.shape == (len(x),)
    for i in range(x.shape[1]):
        assert (i, i) in Q_dict
