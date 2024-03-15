# This script shows how one can use the Deep Galerkin method to solve a
# one-dimensional heat equation (PDE) using deep neural networks.
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
    N = 40      # Number of discretization nodes
    iters = 15000   # Number of learning iterations

    # Define the neural network
    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=128,
              num_layers=3).to(device)

    parser = argparse.ArgumentParser(
                    prog="NeuralFieldsDNNSolver",
                    description="DNN solver for linear first order ODE",
                    epilog="-")

    parser.add_argument('--solve',
                        action="store_true",)
    parser.add_argument('--plot',
                        action="store_true")
    parser.add_argument('--savefig',
                        action="store_true")
    args = parser.parse_args()

    if args.solve:
        # Approximate solution using DGM
        nnet, loss_dgm = minimize_loss_dgm(net,
                                           iterations=iters,
                                           batch_size=64,
                                           lrate=1e-4,
                                           )
        y_dgm = gridEvaluation(nnet, nodes=N)
        np.save("temp_results/heat_sol_1d_dgm", y_dgm)
        np.save("temp_results/heat_sol_1d_dgm_loss", np.array(loss_dgm))

        # Exact solution
        y_exact = exact_solution(k=1, nodes=N)
        np.save("temp_results/heat_sol_exact_1d", y_exact)

    if args.plot:
        y_dgm = np.load("temp_results/heat_sol_1d_dgm.npy")
        loss_dgm = np.load("temp_results/heat_sol_1d_dgm_loss.npy")
        y_exact = np.load("./temp_results/heat_sol_exact_1d.npy")

        mae_dgm = mean_absolute_error(y_exact, y_dgm)

        fig = plt.figure(figsize=(20, 5))
        fig.subplots_adjust(bottom=0.11)
        ax = fig.add_subplot(131)
        im = ax.imshow(y_exact, origin='lower', vmin=0.0, vmax=1.0)
        plt.colorbar(im)
        ax.set_xticks([0, 20, 39])
        ax.set_xticklabels(['0', r'$\frac{\pi}{{\bf 2}}$', r'$\pi$'],
                           fontsize=14, weight='bold')
        ax.set_yticks([0, 20, 39])
        ax.set_yticklabels(['0', '1.5', '3'], fontsize=14, weight='bold')
        ax.title.set_text('Exact solution')
        ax.set_ylabel("Time", fontsize=16, weight='bold')
        ax.set_xlabel("Space", fontsize=16, weight='bold')
        ax.text(0, 40, 'A',
                ha='left',
                fontsize=18,
                weight='bold')

        ax = fig.add_subplot(132)
        im = ax.imshow(y_dgm, origin="lower", vmin=0.0, vmax=1.0)
        plt.colorbar(im)
        ax.set_yticks([0, 20, 39])
        ax.set_yticklabels(['0', '1.5', '3'], fontsize=14, weight='bold')
        ax.set_xticks([0, 20, 39])
        ax.set_xticklabels(['0', r'$\frac{\pi}{{\bf 2}}$', r'$\pi$'],
                           fontsize=14, weight='bold')
        ax.set_xlabel("Space", fontsize=16, weight='bold')
        ax.title.set_text('Approximated solution (DNN)')
        ax.text(0, 40, 'B',
                ha='left',
                fontsize=18,
                weight='bold')

        ax = fig.add_subplot(133)
        ax.plot(np.array(loss_dgm), lw=2.0)
        ax.set_xlabel("Iterations", fontsize=16, weight='bold')
        ax.set_ylabel("Loss", fontsize=16, weight='bold')
        ax.set_xticks([0, int(iters//2), iters])
        ax.set_xticklabels(['0', '7500', '15000'], fontsize=14, weight='bold')
        ticks = np.round(ax.get_yticks(), 2)
        ax.set_yticklabels(ticks, fontsize=14, weight='bold')
        ax.text(0, 4.3, 'C',
                ha='left',
                fontsize=18,
                weight='bold')
        ax.text(9000, 2.5, "DGM MAE: "+str(np.round(mae_dgm, 4)),
                fontsize=13,
                weight='bold')

        if args.savefig:
            plt.savefig("figs/heat_1dim_solution.pdf")
    plt.show()
