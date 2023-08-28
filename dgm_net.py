from torch import nn


class DGMLayer(nn.Module):
    def __init__(self, input_dim=1, hidden_size=50):
        super().__init__()

        self.Z_wg = nn.Linear(hidden_size, hidden_size)
        self.Z_ug = nn.Linear(input_dim, hidden_size, bias=False)

        self.G_wz = nn.Linear(hidden_size, hidden_size)
        self.G_uz = nn.Linear(input_dim, hidden_size, bias=False)

        self.R_wr = nn.Linear(hidden_size, hidden_size)
        self.R_ur = nn.Linear(input_dim, hidden_size, bias=False)

        self.H_wh = nn.Linear(hidden_size, hidden_size)
        self.H_uh = nn.Linear(input_dim, hidden_size, bias=False)

        self.sigma = nn.Tanh()

    def forward(self, x, s_old):
        Z = self.sigma(self.Z_wg(s_old) + self.Z_ug(x))
        G = self.sigma(self.G_wz(s_old) + self.G_uz(x))
        R = self.sigma(self.R_wr(s_old) + self.R_ur(x))
        H = self.sigma(self.H_wh(s_old * R) + self.H_uh(x))
        s_new = (1 - G) * H + Z * s_old
        return s_new


class DGM(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_size=1, num_layers=1):
        super(DGM, self).__init__()
        self.S_in = nn.Linear(input_dim, hidden_size)

        self.layers = nn.ModuleList([DGMLayer(input_dim, hidden_size)
                                     for _ in range(num_layers)])

        self.S_out = nn.Linear(hidden_size, output_dim)

        self.sigma = nn.Tanh()

    def forward(self, x):
        out = self.sigma(self.S_in(x))

        for layer in self.layers:
            out = layer(x, out)

        out = self.S_out(out)

        return out
