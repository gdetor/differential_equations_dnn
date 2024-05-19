# This script shows how to use Ray Tune to search for hyperparameters of the
# neural networks used by the Deep Galerkin method to solve the
# one-dimensional heat equation.
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
from ray import tune
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

import torch

from neural_networks import MLP
from auxiliary_funs import fn_timer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def objectiveRay(config):
    """! This is the objective function which is optimized from the Ray Tuner
    class. For each set of parameters, the neural network is being trained and
    the training loss is being reported to the Tuner.

    @param config A Python dictionary that represents the search space
    (hyperparameters of the neural network and Deep Galerkin method).

    @return Void
    """
    # Define the neural network
    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=128,
              num_layers=3).to(device)

    # Train the neural network to approximate the DE
    _, loss_dgm = minimize_loss_dgm(net,
                                    iterations=config["n_iters"],
                                    batch_size=config["batch_size"],
                                    lrate=config["lrate"],
                                    )

    # Report loss
    session.report({"loss": loss_dgm[-1]})


def optimizeHeat():
    """! This is the main function that instanstiates and calls the Ray Tuner
    class for searching the optimal hyperparameters. The Optuna algorithm is
    used to optimize the objective function.

    @param void

    @return A Python dictionary that containst the optimal hyperparameters for
    solving the one-dimensional heat equation using DNN and the Deep Galerkin
    method.
    """

    # Define the search space of hyperparameters
    search_space = {"batch_size": tune.randint(lower=1, upper=512),
                    "n_iters": tune.randint(1000, 50000),
                    "lrate": tune.loguniform(1e-4, 1e-1),
                    }

    # Set the Optuna optimization algorithm, and the scheduler
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=5)
    scheduler = AsyncHyperBandScheduler()

    # Instantiate the Tuner class
    tuner = tune.Tuner(
                tune.with_resources(
                    objectiveRay,
                    resources={"cpu": 10,
                               "gpu": 1}),
                tune_config=tune.TuneConfig(metric="loss",
                                            mode="min",
                                            search_alg=algo,
                                            scheduler=scheduler,
                                            num_samples=10,
                                            ),
                param_space=search_space,
                )
    # Run the optimization
    results = tuner.fit()

    # Gather the results
    print(results.get_best_result(metric="loss", mode="min").config)
    _ = results.get_best_result(metric="loss", mode="min")
    return results.get_best_result(metric="loss", mode="min").config


if __name__ == '__main__':
    optimizeHeat()
