# This script shows how to solve numerically a Fredholm integral eqaution using
# the Deep Galerkin method and Deep Neural Networks.
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

from neural_networks import DGM
from auxiliary_funs import fn_timer

style.use('tableau-colorblind10')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = {'font.size': 14,
          }
plt.rcParams.update(params)


def exact_solution(t):
    """!
    Exact solution.
    """
    return 2.0 * np.sin(t)


def dgm_loss_func(net, x, k=50):
    """! This is the right-hand side of the Fredholm equation plus the
    initial conditions. There are no boundary conditions thus we omit that term
    in the loss function. This function relies on the autograd to estimate the
    derivatives of the differential equation.

    @param net A Pytorch neural network object that approximates the solution
    of the Fredholm equation.
    neural network (torch tensor).
    @param x The independet (covariates) variables (spatial and temporal).
    @param k Number (int) of discretization nodes used by Monte Carlo
    integration to estimate the integral of the Fredholm equation.

    @return The loss of the Deep Galerkin method for the neural field.
    """

    # Monte Carlo integration
    dr = np.pi / (2 * k)
    integral = 0.0
    for i in range(k):
        t = np.pi/2.0 * torch.rand_like(x)
        integral += torch.sin(x) * torch.cos(t) * net(t)
    integral *= dr

    yhat = net(x)

    L = ((yhat - torch.sin(x) - integral)**2)
    return torch.mean(L)


@fn_timer
def minimize_loss_dgm(net,
                      y_ic=2.0,
                      iterations=1000,
                      batch_size=32,
                      lrate=1e-4,
                      ):
    """! Main loss minimization function. This function implements the Deep
    Galerkin Method.

    @param net A torch neural network that will approximate the solution of
    neural fields.
    @param y_ic Initial conditions value (float) y(0) = y_ic.
    @param iterations Number of learning iterations (int).
    @param batch_size The size of the minibatch (int).
    @param lrate The learning rate (float) used in the optimization.

    @return A torch neural network (trained), and the training loss (list)
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lrate)

    train_loss = []
    for i in range(iterations):
        t = np.pi/2.0 * torch.rand([batch_size, 1])
        t.requires_grad = True
        t = t.to(device)

        optimizer.zero_grad()

        loss = dgm_loss_func(net, t)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if i % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iteration: {i}, Loss: {loss.item()}, LR: {lr}")

    return net, train_loss


def gridEvaluation(net, nodes=10, y_ic=2.0):
    """! Evaluates a torch neural network on a rectangular grid (t, x).)

    @param net A torch neural network object.
    @param nodes Number of spatial discretization nodes for the interval
    [0, pi/2].
    @param y_ic Initial conditions (t=0).

    @return A Python list that contains solution evaluated on the nodes of a
    rectangular grid (t, x).
    """
    t_grid = np.linspace(0, np.pi/2.0, nodes)
    sol = np.zeros((nodes,))
    for i, t in enumerate(t_grid):
        x = torch.ones([1]) * t
        x = x.to(device)
        y = net(x)
        sol[i] = y[0, 0].detach().cpu().numpy()
    return sol


if __name__ == "__main__":
    N = 50      # Number of discretization nodes
    n_iters = 3000    # Number of learning iterations
    batch_size = 32

    parser = argparse.ArgumentParser(
                    prog="NeuralFieldsDNNSolver",
                    description="DNN solver for Fredholm equations",
                    epilog="-")

    parser.add_argument('--solve',
                        action="store_true",)
    parser.add_argument('--plot',
                        action="store_true")
    parser.add_argument('--savefig',
                        action="store_true")
    parser.add_argument('--niters',
                        type=int,
                        default=3000)
    parser.add_argument('--nnodes',
                        type=int,
                        default=50)
    parser.add_argument('--batch-size',
                        type=int,
                        default=32)
    args = parser.parse_args()

    N = args.nnodes
    n_iters = args.niters
    batch_size = args.batch_size

    # Define the neural network
    net = DGM(input_dim=1, output_dim=1, hidden_size=batch_size).to(device)

    if args.solve:
        # Approximate solution using DGM
        nnet, loss_dgm = minimize_loss_dgm(net,
                                           y_ic=2.0,
                                           iterations=n_iters,
                                           batch_size=batch_size,
                                           lrate=1e-4,
                                           )
        y_dgm = gridEvaluation(nnet, nodes=N)
        np.save("./temp_results/fredholm_approx", y_dgm)
        np.save("./temp_results/fredholm_loss", loss_dgm)

    # Exact solution
    t = np.linspace(0, np.pi/2, N)
    y_exact = exact_solution(t)

    if args.plot:
        y_dgm = np.load("./temp_results/fredholm_approx.npy")
        loss_dgm = np.load("./temp_results/fredholm_loss.npy")

        # MAE
        mae_dgm = mean_absolute_error(y_exact, y_dgm)
        print(f"MAE: {mae_dgm}")

        fig = plt.figure(figsize=(17, 5))
        fig.subplots_adjust(bottom=0.15)
        ax1 = fig.add_subplot(121)
        ax1.plot(t, y_exact, label="Exact solution")
        ax1.plot(t, y_dgm, '-x', label="DGM NN solution", ms=5)
        ax1.set_ylim([0, 2.5])
        ax1.set_xticks([0, np.pi/4, np.pi/2])
        ax1.set_xticklabels([r'${\bf 0}$',
                             r'${\bf \frac{\pi}{4}}$',
                             r'${\bf \frac{\pi}{2}}$'],
                            fontsize=18, weight='bold')
        ticks = np.round(ax1.get_yticks(), 2)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
        ax1.legend(fontsize=12)
        ax1.set_xlabel(r"Time (t)", fontsize=14, weight='bold')
        ax1.set_ylabel(r"y(t)", fontsize=14, weight='bold')
        ax1.text(0, 2.71, 'A',
                 va='top',
                 fontsize=18,
                 weight='bold')

        ax2 = fig.add_subplot(122)
        ax2.plot(loss_dgm, label="DGM loss")
        ax2.legend(fontsize=12)
        # ax2.set_ylim([-0.1, 0.5])
        ax2.set_xticks([0, n_iters//2, n_iters])
        ax2.set_xticklabels(['0', str(n_iters//2), str(n_iters)],
                            fontsize=14,
                            weight='bold')
        ticks = np.round(ax2.get_yticks(), 2)
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(ticks, fontsize=14, weight='bold')
        ax2.set_xlabel("Iterations", fontsize=14, weight='bold')
        ax2.set_ylabel("Loss", fontsize=14, weight='bold')
        ax2.text(0, 0.87, 'B',
                 va='top',
                 fontsize=18,
                 weight='bold')

        ax2.text(2000, 0.3,
                 "DGM MAE: "+str(np.round(mae_dgm, 4)),
                 fontsize=11,
                 weight="bold")

        if args.savefig:
            plt.savefig("figs/fredholm_solution.pdf")
    plt.show()
