import torch.nn as nn
import torch


class FourierPulse(nn.Module):
    def __init__(self, length=101, tmin=0, tmax=1):
        super().__init__()
        self.tpulse = tmax - tmin
        p = 1e-2 * torch.randn(length)
        weights = torch.exp(-0.1 * torch.abs(torch.arange(0, length)))
        p = p * weights
        self.params = torch.nn.Parameter(p)
        self.k = torch.arange(length, requires_grad=False).unsqueeze(0)

    def forward(self, x):
        y_sin = torch.sum(self.params * torch.sin(torch.pi * self.k * x / self.tpulse), dim=-1, keepdim=True)
        return y_sin

    def to(self, device):
        self.k = self.k.to(device)
        return super().to(device)


class FourierSeries(nn.Module):
    def __init__(self, n_coeffs=101, output_dim=3, tmin=0, tmax=1):
        super().__init__()
        self.pulses = nn.ModuleList([FourierPulse(length=n_coeffs, tmin=tmin, tmax=tmax) for _ in range(output_dim)])

    def forward(self, x):
        out = torch.cat([pulse(x) for pulse in self.pulses], dim=-1)
        return out

    def to(self, device):
        for pulse in self.pulses:
            pulse.to(device)
        return super().to(device)


class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.uniform_(layer.bias, -0.1, 0.1)

    def forward(self, x):
        output = self.model(x)
        return output


class SIREN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, omega_0=6):
        super(SIREN, self).__init__()
        self.omega_0 = omega_0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sin(self.omega_0 * layer(x))
        return self.final_layer(x)


class RBFN(nn.Module):
    def __init__(self, num_centers=10, output_dim=1):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, 1))
        self.linear = nn.Linear(num_centers, output_dim)

    def forward(self, x):
        rbf = torch.exp(-torch.cdist(x, self.centers) ** 2)
        return self.linear(rbf)


class FourierMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, num_fourier_features=10):
        super(FourierMLP, self).__init__()
        self.fourier_weights = nn.Parameter(torch.randn(num_fourier_features, input_dim))
        layers = [nn.Linear(num_fourier_features * 2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.matmul(x, self.fourier_weights.T)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.model(x)
