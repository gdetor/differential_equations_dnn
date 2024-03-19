# This script solves a one-dimensional neural field equation with Gaussian
# kernel using the Deep Galerkin method and a deep MLP neural network.
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
import argparse

import numpy as np
import matplotlib.pylab as plt

import torch
from torch import nn
from torch.optim import Adam

from scipy.integrate import odeint

from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import pdist, squareform

from auxiliary_funs import fn_timer


FIELD_SIZE = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """!
    Feed-forward neural network
    """
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 hidden_size=8,
                 num_layers=1,
                 bn_elements=64):
        super(MLP, self).__init__()

        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.bn_in = nn.BatchNorm1d(FIELD_SIZE)

        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(num_layers)])

        self.bn_h = nn.ModuleList([nn.BatchNorm1d(FIELD_SIZE)
                                   for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # self.act = nn.ReLU()
        # self.act = myReLU()
        self.act = nn.Tanh()

        self.reset()

    def forward(self, x):
        out = self.act(self.fc_in(x))
        # out = self.bn_in(out)
        for i, layer in enumerate(self.layers):
            out = self.act(layer(out))
            # out = self.bn_h[i](out)
        out = self.fc_out(out)
        return out

    def reset(self):
        nn.init.xavier_uniform_(self.fc_in.weight,
                                gain=nn.init.calculate_gain('tanh'))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc_out.weight)


def z(x):
    """!
    Mexican hat function
    """
    return 4.0 * torch.exp(-0.5 * x**2) - 1.5 * torch.exp(-0.5*x**2/4.5**2)


def heaviside(x):
    """!
    Heaviside function
    """
    return (x > 0) * 1.0 + (x <= 0) * 0.0


def dgm_loss_func(y, y0, x, x0, W, h, S, dr):
    """! This is the right-hand side of the neural field equation plus the
    initial conditions. There are no boundary conditions thus we omit that term
    in the loss function. This function relies on the autograd to estimate the
    derivatives of the differential equation.

    @param y The approximated solution of the differential equation by the
    neural network (torch tensor).
    @param y0 The approximated solution at t = 0 (torch tensor).
    @param x The independet (covariates) variables (spatial and temporal).
    @param x0 The initial values of the independent variables (torch tensor).
    @param W The connectivity matrix of the neural fields equation.
    @param h Is the resting potential of the neural field.
    @param S Is the external input to the neural field (torch tensor).
    @param dr Is the spatial measure of the space Omega, where the neural
    field's integral is computed on.

    @return The loss of the Deep Galerkin method for the neural field.
    """
    Dt = torch.autograd.grad(y,
                             x,
                             grad_outputs=torch.ones_like(y),
                             create_graph=True,
                             retain_graph=True,
                             )[0][:, :, 1].unsqueeze(2)

    # conv = torch.matmul(W * dr, F.sigmoid(y)).detach()
    conv = torch.matmul(W * dr, heaviside(y)).detach()
    L1 = torch.mean((Dt + y - conv - h - S)**2)
    L2 = torch.mean((y0 - x0)**2)
    return L1 + L2


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
    # Set the optimizer
    optimizer = Adam(net.parameters(), lr=lrate)

    # Initialize the spatial dimensions tensor
    R = torch.linspace(-20.0, 20.0, FIELD_SIZE).reshape(1, -1, 1)
    R = torch.tile(R, (batch_size, 1, 1))

    # Compute the distances used in the kernel of the neural field equation
    dr = 40.0 / FIELD_SIZE
    D = torch.cdist(R, R, p=1)
    W = z(D)
    W = W.to(device)

    # Initialize the temporal tensor
    T = torch.linspace(0, 100, steps=batch_size).reshape(-1, 1, 1)
    T = torch.tile(T, (1, FIELD_SIZE, 1))

    # Concatenate the spatial and temporal tensors into X
    X = torch.cat([R, T], dim=2)
    X.requires_grad = True
    X = X.to(device)

    # Set the initial conditions
    T0 = torch.zeros([batch_size, FIELD_SIZE, 1])
    X0 = torch.cat([R, T0], dim=2)
    X0 = X0.to(device)

    X_IC = -1.5 * torch.ones([batch_size, FIELD_SIZE, 1], device=device)
    h = -0.5 * torch.ones([batch_size, FIELD_SIZE, 1], device=device)

    # Set the external input tensor
    # S = torch.zeros([batch_size, FIELD_SIZE, 1], device=device)
    S = torch.exp(-0.5 * R**2)
    S = S.to(device)

    # Main optimization loop
    train_loss = []
    for i in range(iterations):
        optimizer.zero_grad()

        y = net(X)
        y0 = net(X0)

        loss = dgm_loss_func(y, y0, X, X_IC, W, h, S, dr)
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())

        # Learning rate schedule
        if i % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iteration: {i}, Loss: {train_loss[-1]}, LR: {lr}")

        if i > 1000 and i < 3000:
            optimizer.param_groups[0]['lr'] = 1e-4

        if i > 3000:
            optimizer.param_groups[0]['lr'] = 1e-5

    return net, train_loss


def gridEvaluation(net, nodes=32):
    """! Evaluates a torch neural network on a rectangular grid (t, x).)

    @param net A torch neural network object.
    @param nodes Number of spatial discretization nodes for the interval
    [0, 100] x [-20, 20].

    @return A Python list that contains solution evaluated on the nodes of a
    rectangular grid (t, x).
    """
    net.eval()

    t_grid = torch.linspace(0.0, 100.0, steps=nodes).reshape(-1, 1, 1)
    t_grid = torch.tile(t_grid, (1, FIELD_SIZE, 1))
    r_grid = torch.linspace(-20.0, 20.0, steps=FIELD_SIZE).reshape(1, -1, 1)
    r_grid = torch.tile(r_grid, (nodes, 1, 1))

    x = torch.cat([r_grid, t_grid], dim=2)
    x = x.to(device)
    y = net(x)
    return y[:, :, 0].detach().cpu().numpy()


def Sigmoid(x, x0=0.0, theta=1.0):
    """!
    Sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-theta*(x - x0)))


def K(x):
    """!
    Connection intensity function - Mexican hat
    """
    return 4.0 * np.exp(-0.5 * x**2) - 1.5 * np.exp(-0.5*x**2/4.5**2)


def Gaussian(x):
    """!
    Gaussian function
    """
    return np.exp(-0.5*x**2)


def fun(s, t, k, D, d, h):
    """! Right hand side of the neural field equation. Use it only with odeint.
    """
    x = s
    dr = 40 / k
    dx = [-x[i] + np.dot(K(D)[i], heaviside(x)) * dr + h + Gaussian(d)[i]
          for i in range(len(x))]
    return dx


if __name__ == '__main__':
    n_iters = 50000
    batch_size = 100
    t_nodes = 150
    parser = argparse.ArgumentParser(
                    prog="NeuralFieldsDNNSolver",
                    description="DNN solver for 1D neural field equations",
                    epilog="-")

    parser.add_argument('--solve',
                        action="store_true",)
    parser.add_argument('--plot',
                        action="store_true")
    parser.add_argument('--savefig',
                        action="store_true")
    parser.add_argument('--niters',
                        type=int,
                        default=50000)
    parser.add_argument('--nnodes',
                        type=int,
                        default=150)
    parser.add_argument('--batch-size',
                        type=int,
                        default=100)
    args = parser.parse_args()

    t_nodes = args.nnodes
    n_iters = args.niters
    batch_size = args.batch_size

    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=80,
              num_layers=4).to(device)

    if args.solve:
        # Solve Neural Fields equation using a Deep Neural Network
        net, loss = minimize_loss_dgm(net,
                                      iterations=n_iters,
                                      batch_size=batch_size,
                                      lrate=1e-2,
                                      )
        # Evaluate the neural network
        y_dgm = gridEvaluation(net, nodes=t_nodes)

        # Solve using a forward Euler's method and convolutions
        t = np.linspace(0, 100.0, t_nodes)
        d = np.linspace(-20.0, 20.0, FIELD_SIZE)
        D = squareform(pdist(d.reshape(-1, 1), lambda u, v: (u - v).sum()))
        x0 = [-1.5 for _ in range(FIELD_SIZE)]
        y_exact = odeint(fun, x0, t, (FIELD_SIZE, D, d, -0.5))

        torch.save(net, "./temp_results/field_model.pt")
        np.save("./temp_results/field_exact_solution", y_exact)
        np.save("./temp_results/field_dgm_solution", y_dgm)
        np.save("./temp_results/field_exact_solution", y_exact)
        np.save("./temp_results/field_dgm_loss", np.array(loss))

    if args.plot:
        # Load the solutions and the loss
        y_dgm = np.load("./temp_results/field_dgm_solution.npy")
        loss = np.load("./temp_results/field_dgm_loss.npy")
        y_exact = np.load("./temp_results/field_exact_solution.npy")

        # Estimate the mean absolute error
        mae_dgm = mean_absolute_error(y_exact, y_dgm)
        print(mae_dgm)

        fig = plt.figure(figsize=(17, 5))
        fig.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(131)
        ax1.plot(y_exact[-1, :], '--', label="Exact solution", zorder=10, lw=2)
        ax1.plot(y_dgm[-1, :], label="DGM NN solution", ms=5, zorder=0, lw=2)
        ax1.set_xticks([0, FIELD_SIZE//2, FIELD_SIZE])
        ax1.set_xticklabels(['-20', '0', '20'], fontsize=14, weight='bold')
        ax1.set_ylim([-5, 5])
        ax1.axhline(-0.5, c='g', lw=1)
        ticks = np.round(ax1.get_yticks(), 2)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
        ax1.set_xlabel(r"Space (r)", fontsize=14, weight='bold')
        ax1.set_ylabel(r"${\bf u(r, t)}$", fontsize=14, weight='bold')
        ax1.text(0, 6.9, 'A',
                 va='top',
                 fontsize=18,
                 weight='bold')

        ax1 = fig.add_subplot(132)
        ax1.plot(y_exact.mean(axis=1), '--', label="Exact solution", zorder=10,
                 lw=2)
        ax1.plot(y_dgm.mean(axis=1),
                 label="DGM NN solution",
                 ms=5,
                 zorder=0,
                 lw=2)
        ax1.set_xticks([0, 75, 150])
        ax1.set_xticklabels(['0', '50', '100'], fontsize=14, weight='bold')
        ticks = np.round(ax1.get_yticks(), 2)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
        ax1.legend(fontsize=12)
        ax1.set_xlabel(r"Time (t)", fontsize=14, weight='bold')
        ax1.set_ylabel(r"Mean spatial activity", fontsize=14, weight='bold')
        ax1.text(0, -0.53, 'B',
                 va='top',
                 fontsize=18,
                 weight='bold')

        ax2 = fig.add_subplot(133)
        ax2.plot(loss[3:], label="DGM loss")
        # ax2.set_ylim([0, 1])
        ax2.legend(fontsize=12)
        ax2.set_xticks([0, n_iters//2, n_iters])
        ticks = ax2.get_xticks()
        ax2.set_xticklabels(ticks, fontsize=14, weight='bold')
        ticks = np.round(ax2.get_yticks(), 2)
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(ticks, fontsize=14, weight='bold')
        ax2.set_xlabel("Iterations", fontsize=14, weight='bold')
        ax2.set_ylabel("Loss", fontsize=14, weight='bold')
        ax2.text(0, 48, 'C',
                 va='top',
                 fontsize=18,
                 weight='bold')

        ax2.text(30000, 5,
                 "DGM MAE: "+str(np.round(mae_dgm, 4)),
                 fontsize=11,
                 weight='bold')

        if args.savefig:
            plt.savefig("figs/field_solution.pdf")
    plt.show()
