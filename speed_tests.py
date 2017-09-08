import numpy as np
from torch.nn import MSELoss
import numpy.linalg as LA
import torch
from torch.autograd import Variable


class PyTorch:
    def __init__(self, A=None, x=None, y=None, S=None, S_svd=None):
        self.A = torch.from_numpy(A)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.S = torch.from_numpy(S)
        self.S_svd = torch.from_numpy(S_svd)
        self.A_v = Variable(self.A, requires_grad=False)
        self.y_v = Variable(self.y, requires_grad=False)
        self.x_v = Variable(self.x, requires_grad=True)

    def sin(self):
        return torch.sin(self.A)

    def matmul(self):
        return self.A @ self.x

    def add(self):
        return self.A + self.A

    def svd(self):
        return torch.svd(self.S_svd, some=True)[1]

    def grad(self):
        mse = MSELoss()
        m = mse.forward(self.A_v @ self.x_v, self.y_v)
        if self.x_v.grad is not None:
            self.x_v.grad.data.zero_()
        m.backward()
        return self.x_v.grad

    def complex_fn(self):
        x = self.x
        A = self.A
        S = self.S
        return np.sin(x) + np.cos(x)*np.sin(S@x) + torch.dot(x, S@x) + np.exp(S@x)

    def cumsum(self):
        return torch.cumsum(self.A, 0)

    def pow(self):
        return torch.pow(torch.abs(self.x), 1.5)



class NumPy:
    def __init__(self, n, seed=42):
        np.random.seed(42)
        self.n = n
        d = 10 * n
        self.x = np.random.randn(n)
        self.A = np.random.randn(d, n)
        self.y = self.A @ self.x + np.random.randn(d)
        self.S = np.random.randn(n, n)
        self.S_svd = np.random.randn(500, 500)

    def matmul(self):
        return self.A @ self.x

    def add(self):
        return self.A + self.A

    def svd(self):
        return np.linalg.svd(self.S_svd, full_matrices=False)[1]

    def grad(self):
        n, d = self.A.shape
        return 2 * self.A.T @ (self.A@self.x - self.y) / n

    def sin(self):
        return np.sin(self.A)

    def cumsum(self):
        return np.cumsum(self.A, axis=0)

    def pow(self):
        return np.float_power(np.abs(self.x), 1.5)

    def complex_fn(self):
        x = self.x
        A = self.A
        S = self.S
        return np.sin(x) + np.cos(x)*np.sin(S@x) + x.T @ S @ x +  np.exp(S@x)
