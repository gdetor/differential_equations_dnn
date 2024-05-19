# This script shows the effect of the batch normalization on the loss
# minimization process when we solve the heat equations.
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
import numpy as np
import matplotlib.pylab as plt
import matplotlib.style as style

# from sklearn.metrics import mean_absolute_error

import torch
from torch import nn

from neural_networks import MLP
from auxiliary_funs import fn_timer

style.use('tableau-colorblind10')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPBNPost(nn.Module):
    """!
    Feed-forward neural network implementation.
    """
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=50,
                 num_layers=1):
        """
        """
        super(MLPBNPost, self).__init__()

        # Input layer
        self.fc_in = nn.Linear(input_dim, hidden_size)

        # Hidden layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(num_layers)])

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # Non-linear activation function
        # self.act = nn.LeakyReLU()
        self.act = nn.ReLU()

        self.bn = nn.BatchNorm1d(hidden_size)

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
        out = self.act(self.fc_in(x))
        out = self.bn(out)
        for i, layer in enumerate(self.layers):
            out = self.act(layer(out))
            out = self.bn(out)
        out = self.fc_out(out)
        return out

    def reset(self):
        """!
        Initialize (reset) the parameters of the MLP using Xavier's uniform
        distribution.
        """
        nn.init.xavier_uniform_(self.fc_in.weight,
                                gain=nn.init.calculate_gain('tanh'))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc_out.weight)


class MLPBNPre(nn.Module):
    """!
    Feed-forward neural network implementation.
    """
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=50,
                 num_layers=1):
        """
        """
        super(MLPBNPre, self).__init__()

        # Input layer
        self.fc_in = nn.Linear(input_dim, hidden_size)

        # Hidden layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(num_layers)])

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # Non-linear activation function
        # self.act = nn.LeakyReLU()
        self.act = nn.ReLU()

        self.bn = nn.BatchNorm1d(hidden_size)

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
        out = self.fc_in(x)
        out = self.act(self.bn(out))
        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = self.act(self.bn(out))
        out = self.fc_out(out)
        return out

    def reset(self):
        """!
        Initialize (reset) the parameters of the MLP using Xavier's uniform
        distribution.
        """
        nn.init.xavier_uniform_(self.fc_in.weight,
                                gain=nn.init.calculate_gain('tanh'))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc_out.weight)


def dgm_loss_func(net, x, x0, xbd1, xbd2, x_bd1, x_bd2):
    """! This is the right-hand side of the heat equation plus the initial and
    the boundary conditions.
    There are no boundary conditions thus we omit that term
    in the loss function. This function relies on the autograd to estimate the
    derivatives of the differential equation.

    @param net A Pytorch neural network object that approximates the solution
    of the heat equation.
    @param y The approximated solution of the differential equation by the
    neural network (torch tensor).
    @param x The independet (covariates) variables (spatial and temporal).
    @param x0 The initial values of the independent variables (torch tensor).
    @param xbd1 Left boundary condition (nodes).
    @param xbd2 Right boundary condition (nodes).
    @param xbd_1 Actual left boundary condition value.
    @param xbd_2 Actual right boundary condition value.

    @return The loss of the Deep Galerkin method for the differential equation.
    """
    kappa = 1.0
    y = net(x)

    dy = torch.autograd.grad(y,
                             x,
                             grad_outputs=torch.ones_like(y),
                             create_graph=True,
                             retain_graph=True)[0]
    dydt = dy[:, 1].unsqueeze(1)
    dydx = dy[:, 0].unsqueeze(1)

    dydxx = torch.autograd.grad(dydx,
                                x,
                                grad_outputs=torch.ones_like(y),
                                create_graph=True,
                                retain_graph=True)[0][:, 0].unsqueeze(1)

    L_domain = ((dydt - kappa * dydxx)**2)

    y0 = net(x0)
    L_init = ((y0 - torch.sin(x0[:, 0].unsqueeze(1)))**2)

    y_bd1 = net(xbd1)
    y_bd2 = net(xbd2)
    L_boundary = ((y_bd1 - x_bd1)**2 + (y_bd2 - x_bd2)**2)
    return torch.mean(L_domain + L_init + L_boundary)


@fn_timer
def minimize_loss_dgm(net,
                      iterations=1000,
                      batch_size=32,
                      lrate=1e-4,
                      ):
    """! Main loss minimization function. This function implements the Deep
    Galerkin Method.

    @param net A torch neural network that will approximate the solution of
    neural fields.
    @param iterations Number of learning iterations (int).
    @param batch_size The size of the minibatch (int).
    @param lrate The learning rate (float) used in the optimization.

    @return A torch neural network (trained), and the training loss (list)
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lrate)

    t0 = torch.zeros([batch_size, 1], device=device)

    xbd1 = torch.zeros([batch_size, 1], device=device)
    xbd2 = torch.zeros([batch_size, 1], device=device)

    train_loss = []
    for i in range(iterations):
        x = torch.pi * torch.rand([batch_size, 1], device=device)
        t = 3.0 * torch.rand([batch_size, 1], device=device)

        X = torch.cat([x, t], dim=1)
        X.requires_grad = True

        X0 = torch.cat([x, t0], dim=1)

        X_BD1 = torch.cat([xbd1, t], dim=1)
        X_BD2 = torch.cat([xbd2, t], dim=1)

        optimizer.zero_grad()

        loss = dgm_loss_func(net, X, X0, X_BD1, X_BD2, xbd1, xbd2)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if i % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iteration: {i}, Loss: {loss.item()}, LR: {lr}")

    return net, train_loss


if __name__ == "__main__":
    n_iters = 15000   # Number of learning iterations
    n_runs = 5  # Number of experiments

    # Define the neural network
    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=128,
              num_layers=3).to(device)

    loss_no_bn = np.zeros((n_runs, n_iters))
    for i in range(n_runs):
        # Approximate solution using DGM
        _, loss = minimize_loss_dgm(net,
                                    iterations=n_iters,
                                    batch_size=64,
                                    lrate=1e-4,
                                    )
        loss_no_bn[i] = loss

    np.save("./temp_results/relu_heat_loss_nobn", loss_no_bn)

    # Define the neural network
    net = MLPBNPre(input_dim=2,
                   output_dim=1,
                   hidden_size=128,
                   num_layers=3).to(device)

    loss_bn_pre = np.zeros((n_runs, n_iters))
    for i in range(n_runs):
        # Approximate solution using DGM
        _, loss = minimize_loss_dgm(net,
                                    iterations=n_iters,
                                    batch_size=64,
                                    lrate=1e-4,
                                    )
        loss_bn_pre[i] = loss

    np.save("./temp_results/relu_heat_loss_bn_pre", loss_bn_pre)

    # Define the neural network
    net = MLPBNPost(input_dim=2,
                    output_dim=1,
                    hidden_size=128,
                    num_layers=3).to(device)

    loss_bn_post = np.zeros((n_runs, n_iters))
    for i in range(n_runs):
        # Approximate solution using DGM
        _, loss = minimize_loss_dgm(net,
                                    iterations=n_iters,
                                    batch_size=64,
                                    lrate=1e-4,
                                    )
        loss_bn_post[i] = loss
    np.save("./temp_results/relu_heat_loss_bn_post", loss_bn_post)

    # loss_no_bn = np.load("./temp_results/relu_heat_loss_nobn.npy")
    # loss_bn_pre = np.load("./temp_results/relu_heat_loss_bn_pre.npy")
    # loss_bn_post = np.load("./temp_results/relu_heat_loss_bn_post.npy")

    # loss_no_bn = np.load("./temp_results/heat_loss_nobn.npy")
    # loss_bn_pre = np.load("./temp_results/heat_loss_bn_pre.npy")
    # loss_bn_post = np.load("./temp_results/heat_loss_bn_post.npy")

    k, m = 1000, 100
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    ax.plot(loss_no_bn.mean(axis=0)[:k],
            lw=2.0,
            label="No Batch Normalization (BN)")
    ax.plot(loss_bn_pre.mean(axis=0)[:k],
            lw=2.0,
            label="BN before activation")
    ax.plot(loss_bn_post.mean(axis=0)[:k],
            lw=2.0,
            label="BN after activation")
    ax.set_xlabel("Iterations", fontsize=16, weight='bold')
    ax.set_ylabel("Loss", fontsize=16, weight='bold')
    ax.set_xticks([0, k//2, k])
    ax.set_xticklabels(['0', str(k//2), str(k)], fontsize=14, weight='bold')
    ticks = np.round(ax.get_yticks(), 2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax.legend(fontsize=13)

    left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(loss_no_bn.mean(axis=0)[:m],
             lw=2.0,
             label="No Batch Normalization (BN)")
    ax2.plot(loss_bn_pre.mean(axis=0)[:m],
             lw=2.0,
             label="BN before activation")
    ax2.plot(loss_bn_post.mean(axis=0)[:m],
             lw=2.0,
             label="BN after activation")
    ax2.set_xlabel("Iterations", fontsize=16, weight='bold')
    ax2.set_ylabel("Loss", fontsize=16, weight='bold')
    ax2.set_xticks([0, m//2, m])
    ax2.set_xticklabels(['0', str(m//2), str(m)], fontsize=14, weight='bold')
    ticks = np.round(ax.get_yticks(), 2)
    ax2.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax2.set_yticklabels(['0', '0.5', '1.0', '1.5'], fontsize=14, weight='bold')

    # plt.savefig("./figs/batchnorm_effect_tanh.pdf")
    plt.show()
