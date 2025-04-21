import torch.nn as nn
import torch
from abc import abstractmethod


def get_model(model_name, **kwargs):
    """Get model by name."""
    model_dict = {"MLP": MLP, "SIREN": SIREN, "FourierSeries": FourierSeries, "RBFN": RBFN, "FourierMLP": FourierMLP}
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} is not supported.")
    return model_dict[model_name](**kwargs).float()


class PulseGradientBase(nn.Module):
    def __init__(self, gradient_scale=200.0, tmin=0, tmax=1):
        super(PulseGradientBase, self).__init__()
        self.gradient_scale = gradient_scale
        self.tmin = tmin
        self.tmax = tmax
        self.name = "no_model_name"

    def model_output_to_pulse_gradient(self, model_output, bdry_scaling=1.0):
        pulse = (model_output[:, 0:1] + 1j * model_output[:, 1:2]) * bdry_scaling
        gradient = self.gradient_scale * model_output[:, 2:]
        return pulse, gradient


class FourierPulse(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, length=101, tmin=0, tmax=1):
        super().__init__(tmin=tmin, tmax=tmax)
        self.tpulse = tmax - tmin
        p = 1e-2 * torch.randn((length, 2))
        weights = torch.exp(-0.1 * torch.abs(torch.arange(0, length)))
        p = p * weights.unsqueeze(1)
        self.params = torch.nn.Parameter(p)
        self.k = torch.arange(length, requires_grad=False).unsqueeze(0)

    def forward(self, x):
        y_sin = torch.sum(self.params[:, 0] * torch.sin(torch.pi * self.k * x / self.tpulse), dim=-1, keepdim=True)
        y_cos = torch.sum(self.params[:, 1] * torch.cos(torch.pi * self.k * x / self.tpulse), dim=-1, keepdim=True)
        return y_sin + y_cos

    def to(self, device):
        self.k = self.k.to(device)
        return super().to(device)


class FourierSeries(PulseGradientBase):
    def __init__(self, n_coeffs=101, output_dim=3, tmin=0, tmax=1, **kwargs):
        super().__init__(tmin=tmin, tmax=tmax)
        self.pulses = nn.ModuleList([FourierPulse(length=n_coeffs, tmin=tmin, tmax=tmax) for _ in range(output_dim)])
        self.name = "FourierSeries"

    def to(self, device):
        for pulse in self.pulses:
            pulse.to(device)
        return super().to(device)

    def forward(self, x):
        scaling = (x - self.tmin) * (self.tmax - x)
        out = torch.cat([pulse(x) for pulse in self.pulses], dim=-1)
        return self.model_output_to_pulse_gradient(out, scaling)


class MLP(PulseGradientBase):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3, num_layers=3, tmin=0, tmax=1, **kwargs):
        super(MLP, self).__init__(tmin=tmin, tmax=tmax)
        self.name = "MLP"
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
        scaling = (x - self.tmin) * (self.tmax - x)
        output = self.model(x)
        return self.model_output_to_pulse_gradient(output, scaling)


class SIREN(PulseGradientBase):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, omega_0=6, tmin=0, tmax=1, **kwargs):
        super(SIREN, self).__init__(tmin=tmin, tmax=tmax)
        self.name = "SIREN"
        self.omega_0 = omega_0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        scaling = (x - self.tmin) * (self.tmax - x)
        for layer in self.layers:
            x = torch.sin(self.omega_0 * layer(x))
        x = self.final_layer(x)
        return self.model_output_to_pulse_gradient(x, scaling)


class RBFN(PulseGradientBase):
    def __init__(self, num_centers=10, center_spacing=1, output_dim=3, tmin=0, tmax=1, **kwargs):
        super(RBFN, self).__init__(tmin=tmin, tmax=tmax)
        self.centers = nn.Parameter(center_spacing * torch.randn(num_centers, 1))
        self.linear = nn.Linear(num_centers, output_dim)
        self.name = "RBFN"

    def forward(self, x):
        rbf = torch.exp(-torch.cdist(x, self.centers) ** 2)
        rbf = self.linear(rbf)
        return self.model_output_to_pulse_gradient(rbf, (x - self.tmin) * (self.tmax - x))


class FourierMLP(PulseGradientBase):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=3,
        num_layers=3,
        num_fourier_features=10,
        frequency_scale=10,
        tmin=0,
        tmax=1,
        **kwargs,
    ):
        super(FourierMLP, self).__init__(tmin=tmin, tmax=tmax)
        self.fourier_weights = nn.Parameter(frequency_scale * torch.randn(num_fourier_features, input_dim))
        layers = [nn.Linear(num_fourier_features * 2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)
        self.name = "FourierMLP"

    def forward(self, x):
        scaling = (x - self.tmin) * (self.tmax - x)
        x = torch.matmul(x, self.fourier_weights.T)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.model_output_to_pulse_gradient(self.model(x), scaling)
