# This script shows the effect of different batch sizes on the loss function
# when we approximate the solution of the heat equation.
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
import pickle

import numpy as np
import matplotlib.pylab as plt
import matplotlib.style as style

from sklearn.metrics import mean_absolute_error

import torch
# from torch import nn

from neural_networks import MLP
from auxiliary_funs import fn_timer

style.use('tableau-colorblind10')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def exact_solution(k=1, nodes=10):
    """!
    Exact solution of the heat equation (see main text).
    """
    sol = np.zeros((nodes, nodes))
    # t_grid = np.linspace(0, np.pi, nodes)
    t_grid = np.linspace(0, 3, nodes)
    x_grid = np.linspace(0, np.pi, nodes)
    for i, t in enumerate(t_grid):
        for j, x in enumerate(x_grid):
            sol[i, j] = np.sin(x) * np.exp(-k*t)
    return sol


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
    u = net(x)

    du = torch.autograd.grad(u,
                             x,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True,
                             retain_graph=True)[0]
    dudt = du[:, 1].unsqueeze(1)
    dudx = du[:, 0].unsqueeze(1)

    dudxx = torch.autograd.grad(dudx,
                                x,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True,
                                retain_graph=True)[0][:, 0].unsqueeze(1)

    L_domain = ((dudt - kappa * dudxx)**2)

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


def gridEvaluation(net, nodes=10):
    """! Evaluates a torch neural network on a rectangular grid (t, x).)

    @param net A torch neural network object.
    @param nodes Number of spatial discretization nodes for the interval
    [0, 3] x [0, pi].

    @return A Python list that contains solution evaluated on the nodes of a
    rectangular grid (t, x).
    """
    t_grid = np.linspace(0, 3.0, nodes)
    x_grid = np.linspace(0, np.pi, nodes)
    sol = np.zeros((nodes, nodes))
    for i, t in enumerate(t_grid):
        for j, x in enumerate(x_grid):
            X = torch.cat([torch.ones([1, 1]) * x,
                           torch.ones([1, 1]) * t], dim=1)
            X = X.to(device)
            y = net(X)
            sol[i, j] = y[0].detach().cpu().numpy()
    return sol


if __name__ == "__main__":
    n_iters = 15000   # Number of learning iterations
    n_runs = 5
    n_batches = 10

    # Define the neural network for the current batch_size
    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=128,
              num_layers=3).to(device)

    # Batch sizes as powers of two
    batch_sizes_list = [2**i for i in range(n_batches+1)]
    loss_per_batch_size = []
    for batch_size in batch_sizes_list:
        print(f"Testing batch size {batch_size}")

        # Run ten times and collect the losses of each run
        running_loss = np.zeros((n_runs, n_iters))
        for i in range(n_runs):
            # Approximate solution using DGM
            nnet, loss_dgm = minimize_loss_dgm(net,
                                               iterations=n_iters,
                                               batch_size=64,
                                               lrate=1e-4,
                                               )
            running_loss[i] = loss_dgm

        loss_per_batch_size.append(running_loss.mean(axis=0))

    with open("./temp_results/losses.pkl", "wb") as f:
        pickle.dump(loss_per_batch_size, f)

    # with open("./temp_results/losses.pkl", "rb") as f:
    #     loss_per_batch_size = pickle.load(f)

    k, m = 1000, 50
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    for i, bs in enumerate(batch_sizes_list):
        ax.plot(loss_per_batch_size[i][:k],
                lw=2.0,
                label="Batch size: "+str(bs))
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
    for i, bs in enumerate(batch_sizes_list):
        ax2.plot(loss_per_batch_size[i][:m],
                 lw=2.0,
                 label="Bastch size: "+str(bs))
    ax2.set_xlabel("Iterations", fontsize=16, weight='bold')
    ax2.set_ylabel("Loss", fontsize=16, weight='bold')
    ax2.set_xticks([0, m//2, m])
    ax2.set_xticklabels(['0', str(m//2), str(m)], fontsize=14, weight='bold')
    ax2.set_yticks([0.0, 0.25, 0.5])
    ax2.set_yticklabels(['0', '0.25', '0.5'], fontsize=14, weight='bold')

    # plt.savefig("./figs/batchsize_effect.pdf")
    plt.show()
