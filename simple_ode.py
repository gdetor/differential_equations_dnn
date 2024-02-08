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


def exact_solution(t):
    return 2.0 * np.exp(-t)


def dgm_loss_func(net, x, x0, x_ic):
    y = net(x)

    dydt = torch.autograd.grad(y,
                               x,
                               grad_outputs=torch.ones_like(y),
                               create_graph=True,
                               retain_graph=True)[0]

    L_domain = ((dydt + y)**2)

    y0 = net(x0)
    L_init = ((y0 - x_ic)**2)
    return torch.mean(L_domain + L_init)


@fn_timer
def minimize_loss_dgm(net,
                      y_ic=2.0,
                      iterations=1000,
                      batch_size=32,
                      lrate=1e-4,
                      ):
    optimizer = torch.optim.Adam(net.parameters(), lr=lrate)

    y_ic = torch.ones([batch_size, 1], device=device) * y_ic
    t0 = torch.zeros([batch_size, 1], device=device)

    train_loss = []
    for i in range(iterations):
        t = 1.01 * torch.rand([batch_size, 1])
        t.requires_grad = True
        t = t.to(device)

        optimizer.zero_grad()

        loss = dgm_loss_func(net, t, t0, y_ic)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if i % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iteration: {i}, Loss: {loss.item()}, LR: {lr}")

    return net, train_loss


def gridEvaluation(net, nodes=10, y_ic=2.0, method="dgm"):
    t_grid = np.linspace(0, 1.0, nodes)
    sol = np.zeros((nodes,))
    for i, t in enumerate(t_grid):
        x = torch.ones([1]) * t
        x = x.to(device)
        if method == "dgm":
            y = net(x)
        else:
            y = y_ic + x * net(x)
        sol[i] = y[0].detach().cpu().numpy()
    return sol


if __name__ == "__main__":
    N = 25
    iters = 5000
    # Define the neural network
    net = MLP(input_dim=1, output_dim=1, hidden_size=32).to(device)

    # Approximate solution using DGM
    nnet, loss_dgm = minimize_loss_dgm(net,
                                       y_ic=2.0,
                                       iterations=iters,
                                       batch_size=64,
                                       lrate=1e-4,
                                       )
    y_dgm = gridEvaluation(nnet, nodes=N)

    # Exact solution
    t = np.linspace(0, 1.0, N)
    y_exact = exact_solution(t)

    # MAE
    mae_dgm = mean_absolute_error(y_exact, y_dgm)

    fig = plt.figure(figsize=(17, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(t, y_exact, label="Exact solution")
    ax1.plot(t, y_dgm, 'x', label="DGM NN solution", ms=15)
    ax1.set_ylim([0, 2.5])
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(['0', '0.5', '1.0'], fontsize=14, weight='bold')
    ticks = np.round(ax1.get_yticks(), 2)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax1.legend(fontsize=12)
    ax1.set_xlabel(r"Time (t)", fontsize=14, weight='bold')
    ax1.set_ylabel(r"y(t)", fontsize=14, weight='bold')
    ax1.text(0, 2.7, 'A',
             va='top',
             fontsize=18,
             weight='bold')

    ax2 = fig.add_subplot(122)
    ax2.plot(loss_dgm[3:], label="DGM loss")
    ax2.set_ylim([0, 10])
    ax2.legend(fontsize=12)
    ax2.set_xticks([0, 2500, 5000])
    ax2.set_xticklabels(['0', '2500', '5000'], fontsize=14, weight='bold')
    ticks = np.round(ax2.get_yticks(), 2)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax2.set_xlabel("Iterations", fontsize=14, weight='bold')
    ax2.set_ylabel("Loss", fontsize=14, weight='bold')
    ax2.text(0, 10.8, 'B',
             va='top',
             fontsize=18,
             weight='bold')

    ax2.text(2500, 8, "DGM MAE: "+str(np.round(mae_dgm, 4)),
             fontsize=13,
             weight='bold')

    plt.savefig("figs/simple_ode_solution.pdf")
    plt.show()
