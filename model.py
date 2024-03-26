import os
import subprocess

import torch.nn

import framework.functions
from framework.layer import *
from framework.functions import sigmoid


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))

    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))

    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)
    # tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    tmp_dir = os.path.join(os.path.curdir, '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    os.remove(graph_path)  # 删除临时文件
    os.removedirs(tmp_dir)  # 删除临时文件夹（必须先是空文件夹）


class Model(Layer):
    def __init(self):
        super().__init__()
        self.train()

    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)

    @staticmethod
    def train():
        train_mode()

    @staticmethod
    def eval():
        test_mode()


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(None, hidden_size)
        self.l2 = Linear(hidden_size, out_size)

    def forward(self, x):
        y = framework.functions.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

class ThreeLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(None, hidden_size)
        self.l2 = Linear(hidden_size, hidden_size)
        self.l3 = Linear(hidden_size, out_size)

    def forward(self, x):
        y = framework.functions.leakyrelu(self.l1(x))
        y = framework.functions.leakyrelu(self.l2(y))
        y = self.l3(y)
        return y

class MLP(Model):
    def __init__(self, fc_output_size, activation=sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_size):
            layer = Linear(None, out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class Encoder(Model):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)

        self.layers = []
        for i in range(N):
            layer = EncoderLayer(d_model, heads, dropout)
            setattr(self, 'Encoder' + str(i), layer)
            self.layers.append(layer)

        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)


class Decoder(Model):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)

        self.layers = []
        for i in range(N):
            layer = EncoderLayer(d_model, heads, dropout)
            setattr(self, 'Encoder' + str(i), layer)
            self.layers.append(layer)

        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, scr_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, scr_mask, trg_mask)

        return self.norm(x)


class Transformer(Model):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, trg_mask, src_mask)
        output = self.out(d_outputs)
        return output
