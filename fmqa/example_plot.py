import numpy as np
import matplotlib.pyplot as plt
import dimod
import fmqa


def two_complement(x, scaling=True):
    """
    2の補数表現を評価する関数
    例 (scaling=False):
      [0,0,0,1] => 1
      [0,0,1,0] => 2
      [0,1,0,0] => 4
      [1,0,0,0] => -8
      [1,1,1,1] => -1
    """
    x = np.asarray(x, dtype=np.int64)
    val, n = 0, len(x)
    for i in range(n):
        val += (1 << (n - i - 1)) * x[i] * (1 if i > 0 else -1)
    return val * (2 ** (1 - n) if scaling else 1)


def main():
    np.random.seed(0)

    # 初期データ
    xs = np.random.randint(2, size=(5, 16))
    ys = np.array([two_complement(x) for x in xs], dtype=float)

    # FMQAモデル作成
    model = fmqa.FMBQM.from_data(xs, ys)

    # Simulated Annealing sampler
    sa_sampler = dimod.samplers.SimulatedAnnealingSampler()

    # 15回更新、毎回3サンプル追加
    for _ in range(15):
        res = sa_sampler.sample(model, num_reads=3)
        new_xs = res.record["sample"].astype(np.int64)  # ここも修正
        new_ys = np.array([two_complement(x) for x in new_xs], dtype=float)

        xs = np.r_[xs, new_xs]
        ys = np.r_[ys, new_ys]

        model.train(xs, ys)

    # README風の図を保存
    plt.figure(figsize=(8, 4.5))
    plt.plot(ys, "o")
    plt.xlabel("Selection ID")
    plt.ylabel("value (scaled)")
    plt.ylim([-1.0, 1.0])
    plt.tight_layout()
    plt.savefig("readme_like_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
