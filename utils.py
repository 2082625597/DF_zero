import time

import tqdm

from framework import Variable
from framework.core import as_array


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def run_time_test(f, n, *inputs):
    f(*inputs)
    f(*inputs)
    f(*inputs)
    st = time.time()
    print('running...')
    for _ in range(n):
        f(*inputs)
    et = time.time()
    print('average-cost-time:{}'.format((et - st)/n))