# Implementation of the Deep Galerkin layer and neural network as it was
# proposed in Sirignano and Spiliopoulos, 2018.
# Copyright (C) 2024  Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from torch import nn


class DGMLayer(nn.Module):
    """Deep Galerkin layer. This class implements a layer of the neural network
    proposed by [Sirignano and Spiliopoulos, 2018]. The layer operates
    similarly to LSTM cells.
    """
    def __init__(self, input_dim=1, hidden_size=50):
        """ Constructor of the DGMClass. Initializes all the necessary
        parameters and torch layers and functions.

        @param input_dim Number of independent variables (covariates) of the
        differential equation (e.g., input_dim=2 if dy(x, t)/dt).
        @param hidden_size Number of units in each hidden layer.

        @return void

        """
        super().__init__()

        self.Z_wg = nn.Linear(hidden_size, hidden_size)
        self.Z_ug = nn.Linear(input_dim, hidden_size, bias=False)

        self.G_wz = nn.Linear(hidden_size, hidden_size)
        self.G_uz = nn.Linear(input_dim, hidden_size, bias=False)

        self.R_wr = nn.Linear(hidden_size, hidden_size)
        self.R_ur = nn.Linear(input_dim, hidden_size, bias=False)

        self.H_wh = nn.Linear(hidden_size, hidden_size)
        self.H_uh = nn.Linear(input_dim, hidden_size, bias=False)

        # Non-linear activation function
        self.sigma = nn.Tanh()

    def forward(self, x, s_old):
        """ Forward method.

        @param x Differential equation's independent variables (tensor of shape
        [*, input_dim]).
        @param s_old Previous state of the differential eqaution (tensor of
        shape [*, input_dim]).

        @return Output tensor of shape [*, hidden_size].
        """
        Z = self.sigma(self.Z_wg(s_old) + self.Z_ug(x))
        G = self.sigma(self.G_wz(s_old) + self.G_uz(x))
        R = self.sigma(self.R_wr(s_old) + self.R_ur(x))
        H = self.sigma(self.H_wh(s_old * R) + self.H_uh(x))
        s_new = (1 - G) * H + Z * s_old
        return s_new


class DGM(nn.Module):
    """ Deep Galrkin Class implements a neural network based on the work of
    [Sirignano and Spiliopoulos, 2018].
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_size=1, num_layers=1):
        """ Constructor method of the class. It initializes all the necessary
        DGM Layers and sets all the parameters.

        @param input_dim Number (int) of independent variables (covariates) of
        the differential equation (e.g., input_dim=2 if dy(x, t)/dt).
        @param output_dim Dimension (int) of the dependent variable of the
        differential equation.
        @param hidden_size Number (int) of units in each hidden layer.
        @param num_layers Number (int) of hidden layers in the neural network.

        @return void

        """
        super(DGM, self).__init__()
        # Input layer
        self.S_in = nn.Linear(input_dim, hidden_size)

        # DGM Layers
        self.layers = nn.ModuleList([DGMLayer(input_dim, hidden_size)
                                     for _ in range(num_layers)])

        # Output layer
        self.S_out = nn.Linear(hidden_size, output_dim)

        # Nonlinearity
        self.sigma = nn.Tanh()

    def forward(self, x):
        """Forward method.

        @param x Differential equation's independent variables (tensor of shape
        [*, input_dim]).

        @return Output tensor (dependent variable, solution of differential
        equation evaluated on the input x) of shape [*, output_dim].
        """
        out = self.sigma(self.S_in(x))

        for layer in self.layers:
            out = layer(x, out)

        out = self.S_out(out)

        return out
