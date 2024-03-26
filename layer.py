import os

import framework.cuda
from framework.core import *
from framework.functions import linear
import numpy as np


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, key, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(key)
        super().__setattr__(key, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)  # 获取变量name和value（Variable）
        array_dict = {key: framework.cuda.as_numpy(param.data) for key, param in params_dict.items() if
                      param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]  # yield暂停处理并返回值

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()


class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='w')

        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, x=None):
        I, O = self.in_size, self.out_size
        w_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        xp = np if x is None else framework.cuda.get_array_module(x)
        if xp != np:
            w_data = xp.asarray(w_data)

        self.W.data = w_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W(x)

        y = linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = framework.utils.pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        if xp != np:
            W_data = xp.asarray(W_data)
        self.W.data = W_data

    def forward(self, x):
        # FIXME(bn的实现，及conv + bn 混合推理)
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = framework.cuda.get_array_module(x)
            self._init_W(xp)

        y = framework.functions_conv.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = framework.utils.pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = framework.cuda.get_array_module(x)
            self._init_W(xp)

        y = framework.functions_conv.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Dropout(Layer):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        return framework.functions.dropout(x, self.dropout)


class PositionalEncoder(Layer):
    def __init__(self, d_model, dropout, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = Dropout(dropout)
        self.pe = Parameter(np.zeros((max_seq_len, d_model)), name='pe', require_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe.data[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                self.pe.data[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

        self.pe.data = self.pe.data.reshape(1, max_seq_len, d_model)

    def forward(self, x):
        x = x * np.sqrt(self.d_model)  # 对于输入的embedded层乘以sqrt(d_model)可以减少embedded的方差
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadAttntion(Layer):
    def __init__(self, heads, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model / heads
        self.h = heads

        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = linear(d_model, d_model)
        self.dropout = Dropout(dropout)

        self.out = linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            batch, rest = mask.shape[0], mask.shape[1:]
            mask = 1 - mask.reshape(batch, 1, *rest)
            scores = scores + mask * 1e9

        scores = framework.functions.softmax(scores, axis=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = matmul(scores, v)

        return output

    def forward(self, q, k, v, mask=None):

        bs = q.shape(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        o = self.attention(q, k, v, mask, self.dropout)

        concat = o.transpose(1, 2).reshape(bs, -1, self.d_model)

        return self.out(concat)


class FeedForward(Layer):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = Linear(d_model, d_ff)
        self.dropout = Dropout(dropout)
        self.linear_2 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(framework.functions.relu(self.linear_1(x)))
        return self.linear_2(x)


class Norm(Layer):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.d_model = d_model

        self.alpha = Parameter(np.ones(self.d_model))
        self.bias = Parameter(np.ones(self.d_model))
        self.eps = eps

    def forward(self, x: Parameter):
        norm = self.alpha * (x - x.mean(axis=-1, keepdims=True)) \
               / (x.std(axis=-1, keepdims=True) + self.eps) + self.bias
        return norm

class Embedder(Layer):
    def __init__(self, vocab_size, d_model, freeze=False):
        super().__init__()
        self.d_model = d_model

        self.dict_params = Parameter(np.random.normal(0, 1, (vocab_size, d_model)), \
                                     require_grad=not freeze)

    def forward(self, inputs):
        output_shape = inputs.shape
        inputs = inputs.reshape(-1)
        output = self.dict_params[inputs, :]
        return output.reshape(*output_shape, self.d_model)

    def load_pretrained(self, weight):
        assert weight.shape == self.dict_params.shape
        self.dict_params.data = weight


class EncoderLayer(Layer):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attn = MultiHeadAttntion(heads, d_model, dropout)
        self.ff = FeedForward(d_model, 4 * d_model, dropout=dropout)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm2(x)
        return x


class DecoderLayer(Layer):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        self.attn1 = MultiHeadAttntion(heads, d_model, dropout)
        self.attn2 = MultiHeadAttntion(heads, d_model, dropout)
        self.ff = FeedForward(d_model, 4 * d_model, dropout=dropout)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        self.dropout_3 = Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        attn_output1 = self.attn1(x, x, x, trg_mask)
        attn_output1 = self.dropout_1(attn_output1)
        x = x + attn_output1
        x = self.norm1(x)

        attn_output2 = self.attn2(x, e_outputs, e_outputs, src_mask)
        attn_output2 = self.dropout_2(attn_output2)
        x = x + attn_output2
        x = self.norm2(x)

        ff_output = self.ff(x)
        ff_output = self.dropout_3(ff_output)
        x = x + ff_output
        x = self.norm3(x)
        return x