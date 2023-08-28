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
    sol = np.zeros((nodes, nodes))
    t_grid = np.linspace(0, np.pi, nodes)
    x_grid = np.linspace(0, np.pi, nodes)
    for i, t in enumerate(t_grid):
        for j, x in enumerate(x_grid):
            sol[i, j] = np.sin(x) * np.exp(-k*t)
    return sol


def dgm_loss_func(net, x, x0, xbd1, xbd2, x_bd1, x_bd2):
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
    t_grid = np.linspace(0, np.pi, nodes)
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
    N = 40
    iters = 15000
    # Define the neural network
    net = MLP(input_dim=2,
              output_dim=1,
              hidden_size=128,
              num_layers=3).to(device)

    # Approximate solution using DGM
    nnet, loss_dgm = minimize_loss_dgm(net,
                                       iterations=iters,
                                       batch_size=64,
                                       lrate=1e-4,
                                       )
    y_dgm = gridEvaluation(nnet, nodes=N)

    # Exact solution
    y_exact = exact_solution(k=1, nodes=N)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    im = ax.imshow(y_exact, origin='lower')
    plt.colorbar(im)
    ax.title.set_text('Exact solution')

    ax = fig.add_subplot(122)
    im = ax.imshow(y_dgm, origin="lower")
    plt.colorbar(im)
    ax.title.set_text('Approximated solution (DNN)')

    # MAE
    # mae_dgm = mean_absolute_error(y_exact, y_dgm)

    # fig = plt.figure(figsize=(17, 5))
    # ax1 = fig.add_subplot(121)
    # ax1.plot(t, y_exact, label="Exact solution")
    # ax1.plot(t, y_dgm, 'x', label="DGM NN solution", ms=15)
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

    plt.savefig("figs/simple_ode_solution.pdf")
    plt.show()
