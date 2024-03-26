import numpy as np
from framework.core import *


gpu_enable = True

try:
    import cupy
    import cupy as cp
    cp.get_default_pinned_memory_pool().free_all_blocks()
except ImportError:
    gpu_enable = False


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('Cupy cannot be loaded. Install Cupy!')
    return cp.asarray(x)
