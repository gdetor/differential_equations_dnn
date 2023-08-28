import numpy as np
import matplotlib.pylab as plt
import matplotlib.style as style

from scipy.integrate import odeint

from sklearn.metrics import mean_absolute_error

import torch
from torch import nn
# from torch import nn

from neural_networks import DGM
from auxiliary_funs import fn_timer

style.use('tableau-colorblind10')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class myReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones([1], requires_grad=True))

    def forward(self, x):
        return (x > 0) * (self.a * x) + (x <= 0) * 0


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
        self.bn_in = nn.BatchNorm1d(32)

        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(num_layers)])

        self.bn_h = nn.ModuleList([nn.BatchNorm1d(32)
                                   for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, output_dim)

        self.act = nn.ReLU()
        # self.act = nn.Tanh()
        # self.act = myReLU()

        self.reset()

    def forward(self, x):
        out = self.act(self.fc_in(x))
        out = self.bn_in(out)
        for i, layer in enumerate(self.layers):
            out = self.act(layer(out))
            out = self.bn_h[i](out)
        out = self.fc_out(out)
        return out

    def reset(self):
        nn.init.xavier_uniform_(self.fc_in.weight,
                                gain=nn.init.calculate_gain('relu'))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_out.weight)


def fun(s, t, alpha, beta, gamma, delta):
    x = s[0]
    y = s[1]

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    return np.array([dx, dy])


def bar(s, t):
    Iext = 0.5
    alpha, beta, tau = 0.7, 0.8, 2.5
    x = s[0]
    y = s[1]

    dx = x - x**3 / 3 - y + Iext
    dy = (x + alpha - beta * y) / tau

    return np.array([dx, dy])


def foo(s, t):
    x = s[0]
    y = s[1]
    k1, k2 = 1., 1.

    dx = -k1 * x
    dy = k1 * x - k2 * y
    dz = k2 * y

    return np.array([dx, dy, dz])


def dgm_loss_func(net,
                  y,
                  y0,
                  x,
                  x_ic):
    Iext = 0.5
    alpha, beta, tau = 0.7, 0.8, 2.5

    X, Y = y[:, 0].unsqueeze(1), y[:, 1].unsqueeze(1)

    dX = torch.autograd.grad(X,
                             x,
                             grad_outputs=torch.ones_like(X),
                             create_graph=True,
                             retain_graph=True)[0]

    dY = torch.autograd.grad(Y,
                             x,
                             grad_outputs=torch.ones_like(Y),
                             create_graph=True,
                             retain_graph=True)[0]

    # dy = torch.autograd.functional.jacobian(lambda x_: net(x_),
    #                                         x,
    #                                         vectorize=True,
    #                                         strategy="forward-mode")
    # dX = torch.diagonal(torch.diagonal(dy, 0, -1), 0)[0].unsqueeze(1)
    # dY = torch.diagonal(torch.diagonal(dy, 1, -1), 0)[0].unsqueeze(1)
    # dZ = torch.diagonal(torch.diagonal(dy, 2, -1), 0)[0].unsqueeze(1)
    Lx = torch.sum((dX + (X**3/3.0 + Y - Iext - X))**2)
    Ly = torch.sum((dY + (beta * Y - alpha - X) / tau)**2)
    L0 = torch.sum((y0 - x_ic)**2)

    # Lx = torch.sum((dX + k1 * X)**2)
    # Ly = torch.sum((dY - k1 * X + k2 * Y)**2)
    # Lz = torch.sum((dZ - k2 * Y)**2)
    # L0 = torch.sum((y0 - x_ic)**2)

    return Lx + Ly + L0


@fn_timer
def minimize_loss_dgm(net,
                      y_ic,
                      iterations=1000,
                      batch_size=32,
                      lrate=1e-4,
                      ):
    optimizer = torch.optim.Adam(net.parameters(), lr=lrate)

    t0 = torch.zeros([batch_size, 1], device=device)

    num_samples = 200
    T = torch.linspace(0.0, 30.0, steps=num_samples)
    prob = torch.tensor([1/num_samples for _ in range(num_samples)])
    # t.requires_grad = True
    # t = t.to(device)

    train_loss = []
    for i in range(iterations):
        # t = 50.01 * torch.rand([batch_size, 1])
        idx = prob.multinomial(num_samples=batch_size, replacement=False)
        t = T[idx].reshape(-1, 1)
        t.requires_grad = True
        t = t.to(device)

        optimizer.zero_grad()

        y = net(t)
        y0 = net(t0)

        loss = dgm_loss_func(net, y, y0, t, y_ic)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if i % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iteration: {i}, Loss: {loss.item()}, LR: {lr}")

        if i > 5000 and i < 35000:
            optimizer.param_groups[0]['lr'] = 1e-4
        elif i > 35000:
            optimizer.param_groups[0]['lr'] = 1e-5

    return net, train_loss


def gridEvaluation(net, nodes=10):
    t_grid = np.linspace(0.0, 30.0, nodes)
    sol = np.zeros((nodes, 2))
    net.eval()
    for i, t in enumerate(t_grid):
        x = torch.ones([1, 1]) * t
        x = x.to(device)
        y = net(x)
        sol[i] = y[0, :].detach().cpu().numpy()
    return sol


if __name__ == "__main__":
    N = 50
    iters = 80000
    batch_size = 100
    net = DGM(input_dim=1,
              output_dim=2,
              hidden_size=128,
              num_layers=3).to(device)

    # Approximate solution using DGM
    # y_ic = torch.zeros([batch_size, 2], device=device)
    # nnet, loss_dgm = minimize_loss_dgm(net,
    #                                    y_ic,
    #                                    iterations=iters,
    #                                    batch_size=batch_size,
    #                                    lrate=1e-2,
    #                                    )
    # y_dgm = gridEvaluation(nnet, nodes=N)
    # np.save("./temp_results/fn_solution_dgm", y_dgm)
    # np.save("./temp_results/fn_loss_dgm", np.array(loss_dgm))
    y_dgm = np.load("./temp_results/fn_solution_dgm.npy")
    loss_dgm = np.load("./temp_results/fn_loss_dgm.npy")

    # Exact solution
    t = np.linspace(0, 30.0, N)
    y_exact = odeint(bar, np.array([0, 0]), t)

    # plt.figure()
    # plt.plot(y_dgm[:, 0], '-x', label="DGM NN solution")
    # plt.plot(y_exact[:, 0], label="Exact solution")
    # plt.legend()

    # plt.figure()
    # plt.plot(y_dgm[:, 1], '-x', label="DGM NN solution")
    # plt.plot(y_exact[:, 1], label="Exact solution")
    # plt.legend()

    # plt.figure()
    # plt.plot(loss_dgm)

    # MAE
    mae_dgm = mean_absolute_error(y_exact, y_dgm)

    fig = plt.figure(figsize=(17, 5))
    fig.subplots_adjust(wspace=0.3)
    ax1 = fig.add_subplot(131)
    ax1.plot(t, y_exact[:, 0], label="Exact solution")
    ax1.plot(t, y_dgm[:, 0], 'x', label="DGM NN solution", ms=15)
    ax1.set_ylim([-2.0, 1.5])
    ax1.set_xticks([0, 15, 30])
    ax1.set_xticklabels(['0', '15', '30'], fontsize=14, weight='bold')
    ticks = np.round(ax1.get_yticks(), 2)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
    # ax1.legend(fontsize=12)
    ax1.set_xlabel(r"Time (t)", fontsize=14, weight='bold')
    ax1.set_ylabel(r"y(t)", fontsize=14, weight='bold')
    ax1.text(0, 1.75, 'A',
             va='top',
             fontsize=18,
             weight='bold')

    ax1 = fig.add_subplot(132)
    ax1.plot(t, y_exact[:, 1], label="Exact solution")
    ax1.plot(t, y_dgm[:, 1], 'x', label="DGM NN solution", ms=15)
    ax1.set_ylim([-.5, 1.8])
    ax1.set_xticks([0, 15, 30])
    ax1.set_xticklabels(['0', '15', '30'], fontsize=14, weight='bold')
    ticks = np.round(ax1.get_yticks(), 2)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax1.legend(fontsize=12)
    ax1.set_xlabel(r"Time (t)", fontsize=14, weight='bold')
    ax1.set_ylabel(r"w(t)", fontsize=14, weight='bold')
    ax1.text(0, 2.17, 'B',
             va='top',
             fontsize=18,
             weight='bold')

    ax2 = fig.add_subplot(133)
    ax2.plot(loss_dgm[3:], label="DGM loss")
    ax2.set_ylim([-1, 8])
    ax2.legend(fontsize=12)
    ax2.set_xticks([0 + i for i in range(iters, 500)])
    ticks = ax2.get_xticks()
    ax2.set_xticklabels(ticks, fontsize=14, weight='bold')
    ticks = np.round(ax2.get_yticks(), 2)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax2.set_xlabel("Iterations", fontsize=14, weight='bold')
    ax2.set_ylabel("Loss", fontsize=14, weight='bold')
    ax2.text(0, 8.6, 'C',
             va='top',
             fontsize=18,
             weight='bold')

    ax2.text(30000, 5, "DGM MAE: "+str(np.round(mae_dgm, 4)), fontsize=11)

    plt.savefig("figs/fitzhugh_nagumo_solution.pdf")
    plt.show()
