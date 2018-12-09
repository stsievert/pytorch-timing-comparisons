import sys
import numpy as np
import time
import pandas as pd

def time_torch(n, max_iter=100):
    x = np.random.randn(n)
    y = torch.Tensor(x)
    data = []
    for _ in range(max_iter):
        start = time.time()
        _ = y.numpy()
        data += [{'time': time.time() - start,
                  'n': n, 'max_iter': max_iter,
                  'library': 'torch'}]
    return data

def time_tf(n, max_iter=100):
    x = np.random.randn(n)
    c = tf.constant(x)
    #session.run(c.initializer)
    _ = c.eval()  # Ignore first, slow step.
    data = []
    for _ in range(max_iter):
        start = time.time()
        _ = c.eval()
        data += [{'time': time.time() - start,
                  'n': n, 'max_iter': max_iter,
                  'library': 'tensorflow'}]
    return data

def time_tf_eager(n, max_iter=100):
    x = np.random.randn(n)
    c = tf.constant(x)
    data = []
    for _ in range(max_iter):
        start = time.time()
        _ = c.numpy()
        data += [{'time': time.time() - start,
                  'n': n, 'max_iter': max_iter,
                  'library': 'tensorflow-eager'}]
    return data

if __name__ == "__main__":
    library = sys.argv[1]
    N = np.logspace(3, 7, num=10, dtype=int)
    
    if library == "torch":
        import torch
        data = [time_torch(n) for n in N]
    elif library == "tf":
        import tensorflow as tf
        with tf.Session() as session:
            data = [time_tf(n) for n in N]
    elif library == "tf-eager":
        import tensorflow as tf
        tf.enable_eager_execution()
        data = [time_tf_eager(n) for n in N]
    else:
        raise ValueError("")
        
    data = sum(data, [])
        
    df = pd.DataFrame(data)
    df.to_csv(f"./data/{library}.csv")