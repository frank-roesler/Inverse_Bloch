import torch.nn as nn
import torch


class MLPWithBoundary(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, left_boundary=0, right_boundary=1):
        super(MLPWithBoundary, self).__init__()
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output # * (x - self.left_boundary) * (self.right_boundary - x)


class SIREN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, omega_0=30):
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
