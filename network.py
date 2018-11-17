import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

class Generator(chainer.Chain):
    def __init__(self, n_in, n_z, n_layer=2, n_hidden=100):
        assert n_layer > 0, "'n_in' must be non-negative integer."

        self.n_layer = n_layer
        self.n_z = n_z
        super(Generator, self).__init__()
        with self.init_scope():
            self.fc_in = L.Linear(None, n_hidden)
            self._forward = ['fc_in']

            for i in range(n_layer - 1):
                name = 'fc_{}'.format(i + 1)
                setattr(self, name, L.Linear(n_hidden, n_hidden))
                self._forward.append(name)

            self.fc_out = L.Linear(n_hidden, n_in)

    def make_z(self, batchsize, n_z):
        return np.random.uniform(-1, 1, (batchsize, self.n_z)).astype(np.float32)

    def forward(self, z, c):
        h = F.concat([z, c])
        for name in self._forward:
            l = getattr(self, name)
            h = F.sigmoid(l(h))
        return self.fc_out(h)

class Discriminator(chainer.Chain):
    def __init__(self, n_in, n_layer=2, n_hidden=100):
        assert n_layer > 0, "'n_in' must be non-negative integer."

        self.n_layer = n_layer
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.fc_in = L.Linear(None, n_hidden)
            self._forward = ['fc_in']

            for i in range(n_layer - 1):
                name = 'fc_{}'.format(i + 1)
                setattr(self, name, L.Linear(n_hidden, n_hidden))
                self._forward.append(name)

            self.fc_out = L.Linear(n_hidden, n_in)

    def forward(self, r, c):
        h = F.concat([r, c])
        for name in self._forward:
            l = getattr(self, name)
            h = F.sigmoid(l(h))
        return self.fc_out(h)