# This script contains Pytorch neural network classes that can be used as PDE
# and ODE solution approximators.
# Copyright (C) 2024  Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
# import numpy as np

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


def selectActivationFunction(name,
                             beta=1.5):
    if name == "relu":
        print("--------- ReLU selected!")
        return nn.ReLU()
    elif name == "sigmoid":
        print("--------- Sigmoid selected!")
        return nn.Sigmoid()
    elif name == "tanh":
        print("--------- Tanh selected!")
        return nn.Tanh()
    elif name == "leaky_relu":
        print("--------- LeakyReLU selected!")
        return nn.LeakyReLU()
    else:
        print("Activation not found!")
        print("--------- ReLU selected!")
        return nn.ReLU()


class DGMLayer(nn.Module):
    """!
    Implementation of a LSTM-like layer for the neural network proposed by
    J. Sirignano and K. Spiliopoulos, "DGM: A deep learning algorithm for
    solving partial differential equations", 2018.
    """
    def __init__(self, input_size=1, output_size=1, func='relu'):
        """! Constructor of the DGMLayer.

        @param input_size The number of expected features in the input x
        @param output_size The number of desired features in the output s_new

        @return void
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Calculate the gain for the Xavier weight initialization
        gain = nn.init.calculate_gain('relu')

        # Initialize all the parameters
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

        # Set the non-linear activation functions
        if func == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()

    def forward(self, x, s):
        """!
        Forward method.

        @param x Input tensor of shape (*, input_size).
        @param s Previous state tensor of shape (*, output_size).

        @return The new state (solution) of the input.
        """
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
    """!
    DGM LSTM-like neural network.
    """
    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 hidden_size=1,
                 num_layers=1,
                 func="relu"):
        super(DGM, self).__init__()
        # Input layer
        self.x_in = nn.Linear(input_dim, hidden_size)

        # DGM layers
        self.dgm1 = DGMLayer(input_dim, hidden_size, func=func)
        self.layers = nn.ModuleList([DGMLayer(input_dim, hidden_size)
                                     for i in range(num_layers)])

        # Output layer
        self.x_out = nn.Linear(hidden_size, output_dim)

        # Non-linear activation function
        if func == "relu":
            self.sigma = nn.ReLU()
        else:
            self.sigma = nn.Tanh()

        # Initialize input and output layers
        xavier_uniform_(self.x_in.weight)
        xavier_uniform_(self.x_out.weight)

    def forward(self, x):
        """!
        Forward method.

        @param x Input tensor of shape (*, input_dim)

        @note The input_dim is the number of covariates (independent variables)
        and output_dim is the number of dependent variables.

        @return s Output tensor of shape (*, output_dim)
        """
        s = self.sigma(self.x_in(x))
        for layer in self.layers:
            s = layer(x, s)
        s = self.x_out(s)
        return s


class MLP(nn.Module):
    """!
    Feed-forward neural network implementation.
    """
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=50,
                 num_layers=1,
                 batch_norm=False,
                 activation="relu"
                 ):
        """
        """
        super(MLP, self).__init__()
        self.activation = activation

        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

            # Input layer
            self.fc_in = nn.Linear(input_dim, hidden_size, bias=False)

            # Hidden layers
            self.layers = nn.ModuleList([nn.Linear(hidden_size,
                                                   hidden_size,
                                                   bias=False)
                                         for _ in range(num_layers)])
        else:
            print("No batch normalization")
            self.bn = nn.Identity()

            # Input layer
            self.fc_in = nn.Linear(input_dim, hidden_size)

            # Hidden layers
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # Non-linear activation function
        # self.act = nn.Tanh()
        self.act = nn.ReLU()
        self.act = selectActivationFunction(activation)

        # Initialize or reset the parameters of the MLP
        self.reset()

    def forward(self, x):
        """!
        Forward method.

        @param x Input tensor of shape (*, input_dim)

        @note The input_dim is the number of covariates (independent variables)
        and output_dim is the number of dependent variables.

        @return out Tensor of shape (*, output_dim)
        """
        out = self.act(self.bn(self.fc_in(x)))
        for i, layer in enumerate(self.layers):
            out = self.act(self.bn(layer(out)))
        out = self.fc_out(out)
        return out

    def reset(self):
        """!
        Initialize (reset) the parameters of the MLP using Xavier's uniform
        distribution.
        """
        if self.activation not in ["relu", "leaky_relu"]:
            nn.init.xavier_uniform_(self.fc_in.weight,
                                    gain=nn.init.calculate_gain(
                                        self.activation)
                                    )
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain(
                                            self.activation)
                                        )
            nn.init.xavier_uniform_(self.fc_out.weight)
        else:
            nn.init.kaiming_uniform_(self.fc_in.weight,
                                     nonlinearity=self.activation)
            for layer in self.layers:
                nn.init.kaiming_uniform_(layer.weight,
                                         nonlinearity=self.activation)
            nn.init.kaiming_uniform_(self.fc_out.weight,
                                     nonlinearity=self.activation)


class ResidualBlock(nn.Module):
    """!
    Linear residual block.
    """
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 running_elems=100,
                 downsample=None):
        super().__init__()
        self.downsample = downsample

        self.fc1 = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                 nn.BatchNorm1d(running_elems))

        self.fc2 = nn.Sequential(nn.Linear(output_dim, output_dim, bias=False),
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
    """!
    Linear ResNet layer.
    """
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
    """!
    Linear ResNet.
    """
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


if __name__ == "__main__":
    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=50,
              num_layers=1,
              batch_norm=False,
              activation="sigmoid")
