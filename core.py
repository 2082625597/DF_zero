import numpy as np
import weakref
import contextlib
import framework

try:
    import cupy as cp

    array_types = (np.ndarray, cp.ndarray)
except ImportError:
    array_types = np.ndarray


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)

def train_mode():
    return using_config('train', True)

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = framework.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def len(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, power, modulo=None):
        return pow(self, power)

    def __getitem__(self, item):
        return get_item(self, item)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def unsqueeze(self, axis: int):
        return unsqueeze(self, axis)

    def squeeze(self, axis: int):
        return squeeze(self, axis)

    def transpose(self, *axis):
        if len(axis) < 1:
            return transpose(self)
        else:
            return transpose(self, axis)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return std(self, axis, keepdims)

    @property
    def T(self):
        return transpose(self)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Variable(' + p + ')'

    def cleargrad(self):
        self.grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = framework.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = framework.cuda.as_cupy(self.data)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 具体的计算通过forward方法进行, forward中的xs都是ndarray
        if not isinstance(ys, tuple):
            ys = (ys,)

        xp = framework.cuda.get_array_module(ys[0].data)
        outputs = [Variable(as_array(y, xp)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


def as_array(x, array_module=np):
    if array_module.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gys):
        x0, x1 = self.inputs
        gx0 = gys * x1
        gx1 = gys * x0
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, xs):
        return -xs

    def backward(self, gys):
        return -gys


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gys):
        gx0 = gys
        gx1 = -gys
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gys):
        x0, x1 = self.inputs
        gx0 = gys / x1
        gx1 = gys * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, xs):
        y = xs ** self.c
        return y

    def backward(self, gys):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gys
        return gx


class Sin(Function):
    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        y = xp.sin(xs)
        return y

    def backward(self, gys):
        x, = self.inputs
        gx = gys * cos(x)
        return gx


class Cos(Function):
    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        y = xp.cos(xs)
        return y

    def backward(self, gys):
        x, = self.inputs
        gx = - gys * sin(x)
        return gx


class Tanh(Function):
    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        y = xp.tanh(xs)
        return y

    def backward(self, gys):
        y = self.outputs[0]()
        gx = gys * (1 - y * y)
        return gx


def add(x0, x1):
    xp = framework.cuda.get_array_module(x0)
    x1 = as_array(x1, xp)
    return Add()(x0, x1)


def mul(x0, x1):
    xp = framework.cuda.get_array_module(x0)
    x1 = as_array(x1, xp)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    xp = framework.cuda.get_array_module(x0)
    x1 = as_array(x1, xp)
    return Sub()(x0, x1)


def rsub(x0, x1):
    xp = framework.cuda.get_array_module(x0)
    x1 = as_array(x1, xp)
    return Sub()(x1, x0)


def div(x0, x1):
    xp = framework.cuda.get_array_module(x0)
    x1 = as_array(x1, xp)
    return Div()(x0, x1)


def rdiv(x0, x1):
    xp = framework.cuda.get_array_module(x0)
    x1 = as_array(x1, xp)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tan(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, xs):
        self.x_shape = xs.shape
        y = xs.reshape(self.shape)
        return y

    def backward(self, gys):
        return reshape(gys, self.x_shape)


class Transpose(Function):
    def __init__(self, axis=None):
        self.axis = axis

    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        y = xp.transpose(xs, self.axis)
        return y

    def backward(self, gy):
        if self.axis is not None:
            xp = framework.cuda.get_array_module(gy)
            tmp_axes = xp.zeros_like(self.axis)
            for i, ax in enumerate(self.axis):
                tmp_axes[ax] = i
            gx = transpose(gy, tmp_axes)
        else:
            gx = transpose(gy)
        return gx


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, xs):
        self.x_shape = xs.shape
        y = xs.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gys):
        if self.axis is None:
            xp = framework.cuda.get_array_module(gys)
            gx = xp.ones(self.x_shape) * gys
        else:
            tmp = list(gys.shape)
            if not self.keepdims:
                tmp.insert(self.axis, 1)
            tmp = tuple(tmp)
            gys = gys.reshape(tmp)
            gx = broadcast_to(gys, self.x_shape)

        return gx


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, xs):
        self.x_shape = xs.shape

        xp = framework.cuda.get_array_module(xs)
        y = xp.broadcast_to(xs, self.shape)
        return y

    def backward(self, gys):
        gx = sum_to(gys, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape):
        if not isinstance(shape, tuple):
            self.shape = (shape,)
        else:
            self.shape = shape

    def forward(self, xs):
        self.x_shape = xs.shape
        if xs.shape == self.shape:
            return xs

        xp = framework.cuda.get_array_module(xs)
        xs = xp.sum(xs, axis=tuple(range(len(xs.shape) - len(self.shape))))
        for dim, n in enumerate(self.shape):
            if n == 1:
                xs = xp.sum(xs, axis=dim, keepdims=True)

        return xs

    def backward(self, gys):
        gx = broadcast_to(gys, self.x_shape)
        return gx


class MatMul(Function):
    def forward(self, xs, w):
        y = xs.dot(w)
        return y

    def backward(self, gys):
        x, W = self.inputs
        gx = matmul(gys, W.T)
        gw = matmul(x.T, gys)
        return gx, gw


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x, axis=None):
    return Transpose(axis)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def matmul(x, w):
    return MatMul()(x, w)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = framework.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class Cat(Function):
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *inputs):
        xp = framework.cuda.get_array_module(inputs[0])
        z = xp.concatenate(inputs, axis=self.axis)
        return z

    def backward(self, gz):
        inputs = self.inputs
        gradients = []
        start_idx = 0

        for x in inputs:
            end_idx = start_idx + x.shape[self.axis]

            indices = [slice(None)] * gz.ndim
            indices[self.axis] = slice(start_idx, end_idx)

            gradients.append(gz[tuple(indices)])

            start_idx = end_idx

        return tuple(gradients)


def cat(inputs, axis=0):
    return Cat(axis=axis)(*inputs)


class Unsqueeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = framework.cuda.get_array_module(x)
        z = xp.expand_dims(x, axis=self.axis)
        return z

    def backward(self, gys):
        return gys.reshape(self.inputs[0].shape)


def unsqueeze(x, axis):
    return Unsqueeze(axis)(x)


class Squeeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = framework.cuda.get_array_module(x)
        z = xp.squeeze(x, axis=self.axis)
        return z

    def backward(self, gys):
        return gys.reshape(self.inputs[0].shape)


def squeeze(x, axis):
    return Squeeze(axis)(x)


class Mean(Function):
    def __init__(self, dims, keep_dim=False):
        super().__init__()
        self.axis = dims
        self.keep_dim = keep_dim

    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        return xp.mean(xs, axis=self.axis, keepdims=self.keep_dim)

    def backward(self, gys):
        x, = self.inputs

        if self.axis:
            n = x.shape[self.axis]
            if not self.keep_dim:
                gy = gys.unsqueeze(self.axis)
            else:
                gy = gys
        else:
            n = x.size
            gy = gys + x * 0

        return gy / n + x * 0


class Std(Function):
    def __init__(self, axis, keep_dim=False):
        super().__init__()
        self.axis = axis
        self.keep_dim = keep_dim

    def forward(self, xs):
        xp = framework.cuda.get_array_module(xs)
        if self.axis:
            n = xs.shape[self.axis]
        else:
            n = xs.size
        return xp.std(xs, axis=self.axis, keepdims=self.keep_dim) * xp.sqrt(n / (n - 1))

    def backward(self, gys):
        x, = self.inputs
        xp = framework.cuda.get_array_module(gys)
        if self.axis:
            n = x.shape[self.axis]
            if not self.keep_dim:
                gy = unsqueeze(gys, self.axis)
                y = unsqueeze(self.outputs[0](), self.axis)
            else:
                gy = gys
                y = self.outputs[0]()
        else:
            n = x.size
            gy = gys + x * 0
            y = self.outputs[0]() + x * 0

        return (x - mean(x, self.axis, keep_dim=True)) / (n - 1) / y * gy


def mean(x, dims, keep_dim=False):
    return Mean(dims, keep_dim)(x)


def std(x, dims, keep_dim=False):
    return Std(dims, keep_dim)(x)


class Parameter(Variable):
    def __init__(self, data, name=None, require_grad=True):
        super().__init__(data, name)
        self.require_grad = require_grad
