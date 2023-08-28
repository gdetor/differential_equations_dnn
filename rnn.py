import numpy as np
import matplotlib.pylab as plt
import matplotlib.style as style

from scipy.integrate import odeint

from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import pdist, squareform

import torch
from torch import nn
import torch.nn.functional as F
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
        self.bn_in = nn.BatchNorm1d(64)

        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(num_layers)])

        self.fc_out = nn.Linear(hidden_size, output_dim)

        # self.act = nn.ReLU()
        self.act = nn.Tanh()
        # self.act = myReLU()

        self.reset()

    def forward(self, x):
        out = self.act(self.fc_in(x))
        for i, layer in enumerate(self.layers):
            out = self.act(layer(out))
        out = self.fc_out(out)
        return out

    def reset(self):
        nn.init.xavier_uniform_(self.fc_in.weight,
                                gain=nn.init.calculate_gain('tanh'))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc_out.weight)


def Heaviside(x):
    """
    Heaviside function
    """
    return (x > 0) * 1.0 + (x <= 0) * 0.0


def Sigmoid(x, x0=0.0, theta=1.0):
    """
    Sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-theta*(x - x0)))


def K(x):
    """
    Connection intensity function - Mexican hat
    """
    return 4.0 * np.exp(-0.5 * x**2) - 1.5 * np.exp(-0.5*x**2/4.5**2)


def G(x):
    """
    Gaussian function
    """
    return np.exp(-0.5*x**2)


def fun(s, t, k, D, d, h):
    x = s
    dr = 40 / k
    dx = [-x[i] + np.dot(K(D)[i], Sigmoid(x)) * dr + h
          for i in range(len(x))]
    # dx = [-x[i] + np.dot(K(D)[i], Sigmoid(x)) * dr + h + G(d)[i]
    #       for i in range(len(x))]
    return dx


def dgm_loss_func(net, y, y0, x, x_ic, W, h, S):
    Dt = torch.autograd.grad(y,
                             x,
                             grad_outputs=torch.ones_like(y),
                             create_graph=True,
                             )[0][:, 1].unsqueeze(1)

    # RHS = torch.matmul(W, heaviside(y)) + h + S
    # RHS = torch.matmul(W, torch.heaviside(y, V).detach()) * dr + h + S

    L1 = (Dt + y - torch.matmul(W, F.sigmoid(y)).squeeze(2) - h - S)**2

    L2 = (x_ic - y0)**2

    return torch.mean(L1 + L2)


def z(x):
    """!
    Mexican hat function
    """
    return 4.0 * torch.exp(-0.5 * x**2) - 1.5 * torch.exp(-0.5*x**2/4.5**2)


@fn_timer
def minimize_loss_dgm(net,
                      iterations=1000,
                      batch_size=32,
                      lrate=1e-4,
                      ):
    activation = "sigmoid"
    source = "none"

    h = -0.5 * torch.ones([batch_size, 1])
    h = h.to(device)
    y_ic = -1.5 * torch.ones([batch_size, 1])
    y_ic = y_ic.to(device)

    V = None
    if activation == "heaviside":
        V = torch.ones([batch_size, 1])
        V = V.to(device)

    num_samples = 600
    T = torch.linspace(0.0, 100.0, steps=num_samples)
    prob = torch.tensor([1/num_samples for _ in range(num_samples)])

    S = torch.zeros([batch_size, 1])
    R = torch.linspace(-20.0, 20.0, steps=batch_size).reshape(-1, 1)
    t0 = torch.zeros([batch_size, 1])

    dr = 40.0 / batch_size
    D = torch.cdist(R.reshape(-1, 1), R.reshape(-1, 1), p=1)
    W = z(D) * dr
    W = torch.tile(W, (batch_size, 1, 1))
    W = W.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lrate)

    train_loss = []
    for i in range(iterations):
        idx = prob.multinomial(num_samples=batch_size, replacement=False)
        t = T[idx].reshape(-1, 1)

        X = torch.cat([R, t], dim=1)

        X.requires_grad = True
        X = X.to(device)

        X0 = torch.cat([R, t0], dim=1)
        X0 = X0.to(device)

        if source == "sigmoid":
            S = torch.exp(-0.5*R**2)  # * (1.0 / (np.sqrt(2.*np.pi)))
        S = S.to(device)

        optimizer.zero_grad()

        y = net(X)
        y0 = net(X0)

        loss = dgm_loss_func(net, y, y0, X, y_ic, W, h, S)

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


def gridEvaluation(net, nodes=32):
    t_grid = np.linspace(0.0, 100.0, nodes)
    r_grid = np.linspace(-20.0, 20.0, nodes)
    sol = np.zeros((nodes, nodes))
    net.eval()
    for i, t in enumerate(t_grid):
        for j, r in enumerate(r_grid):
            x = torch.cat([torch.ones([1, 1]) * r,
                           torch.ones([1, 1]) * t],
                          dim=1)
            x = x.to(device)
            y = net(x)
            sol[i, j] = y[0, 0].detach().cpu().numpy()
    return sol


if __name__ == "__main__":
    N, M = 50, 64
    iters = 15000
    batch_size = 256
    # Define the neural network
    # net = MLP(input_dim=2,
    #           output_dim=1,
    #           hidden_size=128,
    #           num_layers=1).to(device)
    net = DGM(input_dim=2,
              output_dim=1,
              hidden_size=64,
              num_layers=3).to(device)

    # Approximate solution using DGM
    y_ic = torch.zeros([batch_size, M, 1], device=device)
    nnet, loss_dgm = minimize_loss_dgm(net,
                                       iterations=iters,
                                       batch_size=batch_size,
                                       lrate=1e-2,
                                       )
    y_dgm = gridEvaluation(nnet, nodes=M)

    # Exact solution
    t = np.linspace(0, 100.0, N)
    d = np.linspace(-20.0, 20.0, M)
    D = squareform(pdist(d.reshape(-1, 1), lambda u, v: (u - v).sum()))
    x0 = [-1.5 for _ in range(M)]
    y_exact = odeint(fun, x0, t, (M, D, d, -0.5))
    plt.figure()
    plt.plot(y_dgm[-1, :], '-x', label="DGM NN solution")
    plt.plot(y_exact[-1, :], label="Exact solution")
    plt.legend()

    # plt.figure()
    # plt.plot(loss_dgm)
    # MAE
    # mae_dgm = mean_absolute_error(y_exact, y_dgm)
    # mae_trial = mean_absolute_error(y_exact, y_trial)

    # fig = plt.figure(figsize=(17, 5))
    # ax1 = fig.add_subplot(121)
    # for i in range(3):
    #     ax1.plot(t, y_exact[i, :], label="Exact solution")
    # # ax1.plot(t, y_dgm, 'x', label="DGM NN solution", ms=15)
    # # ax1.plot(t, y_trial, 'o', label="Trial NN solution", ms=10)
    # ax1.set_ylim([0, 2.5])
    # ax1.set_xticks([0, 0.5, 1.0])
    # ax1.set_xticklabels(['0', '0.5', '1.0'], fontsize=14, weight='bold')
    # ticks = np.round(ax1.get_yticks(), 2)
    # ax1.set_yticks(ticks)
    # ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
    # ax1.legend(fontsize=12)
    # ax1.set_xlabel(r"Time (t)", fontsize=14, weight='bold')
    # ax1.set_ylabel(r"y(t)", fontsize=14, weight='bold')
    # ax1.text(0, 2.7, 'A',
    #          va='top',
    #          fontsize=18,
    #          weight='bold')

    # ax2 = fig.add_subplot(122)
    # ax2.plot(loss_dgm[3:], label="DGM loss")
    # ax2.plot(loss_trial[3:], label="Trial loss")
    # ax2.set_ylim([0, 10])
    # ax2.legend(fontsize=12)
    # ax2.set_xticks([0 + i for i in range(iters, 500)])
    # ticks = ax2.get_xticks()
    # ax2.set_xticklabels(ticks, fontsize=14, weight='bold')
    # ticks = np.round(ax2.get_yticks(), 2)
    # ax2.set_yticks(ticks)
    # ax2.set_yticklabels(ticks, fontsize=14, weight='bold')
    # ax2.set_xlabel("Iterations", fontsize=14, weight='bold')
    # ax2.set_ylabel("Loss", fontsize=14, weight='bold')
    # ax2.text(0, 10.8, 'B',
    #          va='top',
    #          fontsize=18,
    #          weight='bold')

    # ax2.text(2500, 8, "DGM MAE: "+str(np.round(mae_dgm, 4)), fontsize=11)
    # ax2.text(2500, 8.5, "Trial MAE: "+str(np.round(mae_trial, 4)), fontsize=11)

    # plt.savefig("figs/simple_ode_solution.pdf")
    plt.show()
