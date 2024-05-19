# This script shows how one can approximate a very simple function using a
# feed-forward neural network (showcase of universal approximation theorem).
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

import torch
from torch import nn
from torch.optim import Adam


# Define a simple one-hidden layer feed-forward neural network
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.in2h = nn.Linear(1, 3, bias=False)
        self.h2out = nn.Linear(3, 1)

        self.act = nn.Tanh()

    def forward(self, x):
        out = self.act(self.in2h(x))
        out = self.h2out(out)
        return out


# Define the function we would like to approximate
def f(x):
    return np.sin(3*x)


if __name__ == '__main__':
    n_samples = 50          # Number of data points where to approximate f(x)
    n_iters = 100000        # Number of minimization iterations

    # Evaluate f(x) on grid [-1, 1]
    x0 = np.linspace(-1, 1, n_samples)
    y0 = f(x0)

    # Convert data to Pytorch tensors
    X = torch.from_numpy(x0.astype('f')).reshape(-1, 1)
    Y = torch.from_numpy(y0.astype('f')).reshape(-1, 1)

    # Initialize the neural network, optimizer, and the loss function
    net = Perceptron()
    optimizer = Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Minimize the loss - train the neural network to approximate f(x)
    train_loss = []
    net.train()
    for i in range(n_iters):
        # idx = np.random.randint(0, n_samples)
        # x, y = X[idx].reshape(-1, 1), Y[idx].reshape(-1, 1)

        optimizer.zero_grad()

        yhat = net(X)

        loss = criterion(yhat, Y)
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())
        if i % 5000 == 0:
            print(i, loss.item())

    net.eval()
    with torch.no_grad():
        yhat = []
        for x in X:
            yhat.append(net(x.reshape(-1, 1)).detach().numpy()[0])

    np.save("ground_truth", y0)
    np.save("space", x0)
    np.save("approximation", np.array(yhat))
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    ax.plot(x0, y0, 'o', color='k', label='Ground truth', ms=10)
    ax.plot(x0, yhat, '-', color='orange', label="Approximation", lw=2)

    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['-1', '1'], fontsize=14, weight='bold')
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['-1', '1'], fontsize=14, weight='bold')
    ax.set_xlabel(r"$x$", fontsize=21, weight='bold')
    ax.set_ylabel(r"$f(x)$", fontsize=21, weight='bold')

    plt.savefig("universal_approx.pdf")
    plt.show()
