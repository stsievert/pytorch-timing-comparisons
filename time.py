import sys
import numpy as np
from time import perf_counter
import pandas as pd


def time_torch(n, max_iter=100):
    x = np.random.randn(n)
    y = torch.Tensor(x)
    data = []
    for _ in range(max_iter):
        start = perf_counter()
        _ = y.numpy()
        datum = {
            "time": perf_counter() - start,
            "n": n,
            "max_iter": max_iter,
            "library": "torch",
        }
        data.append(datum)
    return data


def time_tf(n, max_iter=100):
    x = np.random.randn(n)
    c = tf.constant(x)
    # session.run(c.initializer)
    _ = c.eval()  # Ignore first, slow step.
    data = []
    for _ in range(max_iter):
        start = perf_counter()
        _ = c.eval()
        datum = {
            "time": perf_counter() - start,
            "n": n,
            "max_iter": max_iter,
            "library": "tensorflow",
        }
        data.append(datum)
    return data


def time_tf_eager(n, max_iter=100):
    x = np.random.randn(n)
    c = tf.constant(x)
    data = []
    for _ in range(max_iter):
        start = perf_counter()
        _ = c.numpy()
        elapsed = perf_counter() - start
        datum = {
            "time": elapsed,
            "n": n,
            "max_iter": max_iter,
            "library": "tensorflow-eager",
        }
        data.append(datum)
    return data


if __name__ == "__main__":
    library = sys.argv[1]
    N = np.logspace(3, 7, num=10, dtype=int)

    if library == "torch":
        import torch

        data = [time_torch(n) for n in N]
        version = torch.__version__
    elif library == "tf":
        import tensorflow as tf

        with tf.Session() as session:
            data = [time_tf(n) for n in N]
        version = tf.__version__
    elif library == "tf-eager":
        import tensorflow as tf

        tf.enable_eager_execution()
        data = [time_tf_eager(n) for n in N]
        version = tf.__version__
    elif library == "tf2":
        import tensorflow as tf

        data = [time_tf_eager(n) for n in N]
        version = tf.__version__
    else:
        raise ValueError("")

    data = sum(data, [])

    df = pd.DataFrame(data)
    df[f"version"] = version
    df["np_version"] = np.__version__
    df.to_csv(f"./data/{library}.csv")
