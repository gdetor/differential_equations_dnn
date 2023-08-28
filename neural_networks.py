import numpy as np

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class DGMLayer(nn.Module):
    """
    Implementation of a LSTM-like layer for the neural network proposed by
    J. Sirignano and K. Spiliopoulos, "DGM: A deep learning algorithm for
    solving partial differential equations", 2018.
    """
    def __init__(self, input_size=1, output_size=1, bias=True):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        gain = nn.init.calculate_gain('tanh')

        self.Uz = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))
        self.Ug = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))
        self.Ur = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))
        self.Uh = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))

        self.Wz = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))
        self.Wg = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))
        self.Wr = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))
        self.Wh = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))

        self.bz = nn.Parameter(torch.zeros([1, output_size]))
        self.bg = nn.Parameter(torch.zeros([1, output_size]))
        self.br = nn.Parameter(torch.zeros([1, output_size]))
        self.bh = nn.Parameter(torch.zeros([1, output_size]))

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x, s):
        Z = self.act1(torch.matmul(x, self.Uz) + torch.matmul(s, self.Wz) +
                      self.bz)
        G = self.act1(torch.matmul(x, self.Ug) + torch.matmul(s, self.Wg) +
                      self.bg)
        R = self.act1(torch.matmul(x, self.Ur) + torch.matmul(s, self.Wr) +
                      self.br)

        H = self.act2(torch.matmul(x, self.Uh) + torch.matmul(s*R, self.Wh) +
                      self.bh)

        s_new = (torch.ones_like(G) - G) * H + Z * s

        return s_new


class DGM(nn.Module):
    """ DGM neural network (see the reference in DGMLayer)
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_size=1, num_layers=1):
        super(DGM, self).__init__()
        self.x_in = nn.Linear(input_dim, hidden_size)

        self.dgm1 = DGMLayer(input_dim, hidden_size)
        self.layers = nn.ModuleList([DGMLayer(input_dim, hidden_size)
                                     for i in range(num_layers)])

        self.x_out = nn.Linear(hidden_size, output_dim)

        self.sigma = nn.ReLU()

        xavier_uniform_(self.x_in.weight)
        xavier_uniform_(self.x_out.weight)

    def forward(self, x):
        s = self.sigma(self.x_in(x))
        for layer in self.layers:
            s = layer(x, s)
        s = self.x_out(s)
        return s


class MLP(nn.Module):
    """
    Feed-forward neural network
    """
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=50,
                 num_layers=1,
                 bn_elements=64):
        super(MLP, self).__init__()

        self.fc_in = nn.Linear(input_dim, hidden_size)

        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # self.act = nn.LeakyReLU()
        self.act = nn.Tanh()

        self.reset()

    def forward(self, x):
        out = self.act(self.fc_in(x))
        for i, layer in enumerate(self.layers):
            out = self.act(layer(out))
        out = self.fc_out(out)
        return out

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            f_out, f_in = layer.weight.shape[0], layer.weight.shape[1]
            n = f_in * f_out
            std = np.sqrt(2.0 / n)
            w_rnd = torch.randn([f_out, f_in])
            layer.weight.data = w_rnd * std

    def reset(self):
        nn.init.xavier_uniform_(self.fc_in.weight,
                                gain=nn.init.calculate_gain('tanh'))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc_out.weight)


class EnsembleMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=8,
                 n_predictors=32):
        super().__init__()

        self.mini_mlp = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, output_dim))

        self.mlps = nn.ModuleList([self.mini_mlp
                                   for i in range(n_predictors)])

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, output_dim))

        self.fc_out = nn.Linear(n_predictors, n_predictors)

        self.mlps.apply(self.init_weights)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x):
        out_g = self.mlp(x)

        out = torch.cat([mlp_(x[:, i, :])
                         for i, mlp_ in enumerate(self.mlps)], dim=1)
        out = self.fc_out(out)
        return out.unsqueeze(2) + out_g

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # return nn.functional.hardtanh(grad_output)
        return grad_output.clamp_(0, 1)


class HeavisideSTE(nn.Module):
    def __init__(self):
        super(HeavisideSTE, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        ctx.save_for_backward(input)
        return (alpha * input * (input > 0)).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= ctx.alpha
        grad_input[input < 0] = 0.0
        return grad_input.float(), None


class myReLU(nn.Module):
    def __init__(self):
        super(myReLU, self).__init__()
        self.alpha = nn.Parameter(torch.ones([1]))

    def forward(self, x):
        x = ReLUFunction.apply(x, self.alpha)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 running_elems=100,
                 downsample=None):
        super().__init__()
        self.downsample = downsample

        self.fc1 = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(running_elems))

        self.fc2 = nn.Sequential(nn.Linear(output_dim, output_dim),
                                 nn.BatchNorm1d(running_elems))

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out


class ResNetLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=64,
                 running_elements=100,
                 n_blocks=2,
                 block=ResidualBlock):
        super().__init__()

        self.downsample = None
        if input_dim != hidden_size:
            self.downsample = nn.Linear(input_dim, hidden_size, bias=False)

        self.blocks = nn.Sequential(
                block(input_dim=input_dim,
                      output_dim=hidden_size,
                      running_elems=running_elements,
                      downsample=self.downsample),
                *[block(input_dim=hidden_size,
                        output_dim=hidden_size,
                        running_elems=running_elements,
                        downsample=None,
                        ) for _ in range(n_blocks-1)]
                )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_size=32):
        super().__init__()

        self.layer1 = ResNetLayer(input_dim=input_dim,
                                  output_dim=hidden_size,
                                  hidden_size=hidden_size,
                                  running_elements=100,
                                  n_blocks=3)
        self.layer2 = ResNetLayer(input_dim=hidden_size,
                                  output_dim=hidden_size,
                                  hidden_size=hidden_size,
                                  running_elements=100,
                                  n_blocks=3)

        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc_out(out)
        return out


def cyclical_learning_rate(batch_step,
                           step_size,
                           base_lr=0.001,
                           max_lr=0.006,
                           mode='triangular',
                           gamma=0.999995):
    cycle = np.floor(1 + batch_step / (2. * step_size))
    x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)

    lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))

    if mode == 'triangular':
        pass
    elif mode == 'triangular2':
        lr_delta = lr_delta * 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        lr_delta = lr_delta * (gamma**(batch_step))
    else:
        raise ValueError('mode must be "triangular", "triangular2",\
                         or "exp_range"')

    lr = base_lr + lr_delta

    return lr


if __name__ == "__main__":
    x = 2 * torch.rand(5, 1) - 1
    print(x)
    relu = myReLU()
    print(relu(x))
