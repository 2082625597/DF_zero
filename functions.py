import numpy as np

import framework.cuda
from framework.core import *


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gys):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gys * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


class Sigmoid(Function):
    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        self.out = 1 / (1 + xp.exp(-xs))
        return self.out

    def backward(self, gys):
        gx = gys * (1.0 - self.out) * self.out
        return gx


class LeakyRelu(Function):
    def forward(self, xs):
        return np.where(xs > 0, xs, 0.1 * xs)

    def backward(self, gys):
        x, = self.inputs
        gx = np.where(x.data > 0, gys.data, 0.1 * gys.data)
        return gx


class F_Linear(Function):
    def __init__(self):
        self.b_shape = None

    def forward(self, xs, w, b=None):
        xp = framework.cuda.get_array_module(xs)
        y = xp.matmul(xs, w)
        if b is None or len(b) == 0:
            return y
        else:
            self.b_shape = b.shape
            return y + b

    def backward(self, gys):
        x, W, b = self.inputs
        gx = matmul(gys, W.T)
        gw = matmul(x.T, gys)

        if self.b_shape is None:
            xp = framework.cuda.get_array_module(gys)
            return gx, gw, Variable(xp.zeros(0))
        else:
            gb = sum_to(gys, self.b_shape)
            return gx, gw, gb


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def sigmoid(x):
    return Sigmoid()(x)


def leakyrelu(x):
    return LeakyRelu()(x)


def linear(x, w, b=None):
    return F_Linear()(x, w, b)


class Exp(Function):
    def forward(self, x):
        xp = framework.cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = framework.cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = framework.cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


def logsumexp(x, axis=1):
    xp = framework.cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        # convert to one-hot
        xp = framework.cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y, t):
    """
    [WAR] This function is not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


class ReLU(Function):
    def forward(self, x):
        xp = framework.cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if framework.Config.train:
        xp = framework.cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
