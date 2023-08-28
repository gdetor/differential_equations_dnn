import numpy as np
import matplotlib.pylab as plt
import matplotlib.style as style

from sklearn.metrics import mean_absolute_error

import torch
# from torch import nn

from neural_networks import MLP, DGM
from auxiliary_funs import fn_timer

style.use('tableau-colorblind10')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = {'font.size': 14,
          }
plt.rcParams.update(params)


def exact_solution(t):
    return 2.0 * np.sin(t)


def dgm_loss_func(net, x, n=50):
    dr = np.pi / (2 * n)
    f = net(x)

    integral = 0.0
    for i in range(n):
        y = np.pi/2.0 * torch.rand_like(x)
        integral += torch.sin(x) * torch.cos(y) * net(y)
    integral *= dr

    L = ((f - torch.sin(x) - integral)**2)
    return torch.mean(L)


@fn_timer
def minimize_loss_dgm(net,
                      y_ic=2.0,
                      iterations=1000,
                      batch_size=32,
                      lrate=1e-4,
                      ):
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


def gridEvaluation(net, nodes=10, y_ic=2.0, method="dgm"):
    t_grid = np.linspace(0, np.pi/2.0, nodes)
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
    N = 50
    iters = 3000
    # Define the neural network
    # net = MLP(input_dim=1, output_dim=1, hidden_size=32).to(device)
    net = DGM(input_dim=1, output_dim=1, hidden_size=32).to(device)

    # Approximate solution using DGM
    nnet, loss_dgm = minimize_loss_dgm(net,
                                       y_ic=2.0,
                                       iterations=iters,
                                       batch_size=32,
                                       lrate=1e-4,
                                       )
    y_dgm = gridEvaluation(nnet, nodes=N)

    # Exact solution
    t = np.linspace(0, np.pi/2, N)
    y_exact = exact_solution(t)

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
    ax2.set_ylim([0, 0.5])
    ax2.set_xticks([0 + i for i in range(iters, 500)])
    ticks = ax2.get_xticks()
    ax2.set_xticklabels(ticks, fontsize=14, weight='bold')
    ticks = np.round(ax2.get_yticks(), 2)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax2.set_xlabel("Iterations", fontsize=14, weight='bold')
    ax2.set_ylabel("Loss", fontsize=14, weight='bold')
    ax2.text(0, 0.54, 'B',
             va='top',
             fontsize=18,
             weight='bold')

    ax2.text(2000, 0.3, "DGM MAE: "+str(np.round(mae_dgm, 4)), fontsize=11)

    plt.savefig("figs/fredholm_solution.pdf")
    plt.show()
