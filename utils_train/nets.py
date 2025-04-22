import torch.nn as nn
import torch
from abc import abstractmethod


def get_model(model_name, **kwargs):
    """Get model by name."""
    model_dict = {
        "MLP": MLP,
        "SIREN": SIREN,
        "FourierSeries": FourierSeries,
        "RBFN": RBFN,
        "FourierMLP": FourierMLP,
        "MixedModel": MixedModel,
    }
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} is not supported.")
    return model_dict[model_name](**kwargs).float()


class PulseGradientBase(nn.Module):
    def __init__(self, gradient_scale=200.0, tmin=None, tmax=None, output_dim=3, **kwargs):
        super(PulseGradientBase, self).__init__()
        self.gradient_scale = gradient_scale
        self.tmin = tmin
        self.tmax = tmax
        self.output_dim = output_dim
        self.name = "no_model_name"

    def model_output_to_pulse_gradient(self, model_output, x):
        if self.output_dim != 3:
            return model_output
        pulse = model_output[:, 0:1] + 1j * model_output[:, 1:2]
        gradient = self.gradient_scale * model_output[:, 2:]
        if self.tmin is None or self.tmax is None:
            return pulse, gradient
        bdry_scaling = (x - self.tmin) * (self.tmax - x)
        return pulse * bdry_scaling, gradient


class MixedModel(PulseGradientBase):
    def __init__(self, tmin=None, tmax=None, **kwargs):
        super(MixedModel, self).__init__(tmin=tmin, tmax=tmax, **kwargs)
        self.name = "MixedModel"
        self.model1 = FourierPulse(tmin, tmax, **kwargs)
        self.model2 = FourierPulse(tmin, tmax, **kwargs)
        self.model3 = MLP(**kwargs, output_dim=1)

    def to(self, device):
        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)
        self.model3 = self.model3.to(device)
        return super().to(device)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.gradient_scale * self.model3(x)
        pulse = out1 + 1j * out2
        gradient = out3
        return pulse, gradient


class FourierPulse(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, **kwargs):
        super().__init__()
        self.tpulse = t_max - t_min
        p = 1e-3 * torch.randn((2 * n_coeffs + 1, 2))
        weights = torch.exp(-0.01 * (torch.arange(-n_coeffs, n_coeffs + 1)) ** 2)
        p = p * weights.unsqueeze(1)
        self.params = torch.nn.Parameter(p)
        self.k = torch.arange(-n_coeffs, n_coeffs + 1, requires_grad=False).unsqueeze(0)

    def to(self, device):
        self.k = self.k.to(device)
        return super().to(device)

    def forward(self, x):
        y_sin = torch.sum(self.params[:, 0] * torch.sin(torch.pi * self.k * x / self.tpulse), dim=-1, keepdim=True)
        y_cos = torch.sum(self.params[:, 1] * torch.cos(torch.pi * self.k * x / self.tpulse), dim=-1, keepdim=True)
        return y_sin + y_cos


class FourierSeries(PulseGradientBase):
    def __init__(self, tmin, tmax, n_coeffs=101, output_dim=3, **kwargs):
        super().__init__(tmin=tmin, tmax=tmax, output_dim=output_dim**kwargs)
        self.pulses = nn.ModuleList([FourierPulse(tmin, tmax, length=n_coeffs) for _ in range(output_dim)])
        self.name = "FourierSeries"

    def to(self, device):
        for pulse in self.pulses:
            pulse.to(device)
        return super().to(device)

    def forward(self, x):
        out = torch.cat([pulse(x) for pulse in self.pulses], dim=-1)
        return self.model_output_to_pulse_gradient(out, x)


class MLP(PulseGradientBase):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3, num_layers=3, tmin=None, tmax=None, **kwargs):
        super(MLP, self).__init__(tmin=tmin, tmax=tmax, output_dim=output_dim, **kwargs)
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
        output = self.model(x)
        return self.model_output_to_pulse_gradient(output, x)


class SIREN(PulseGradientBase):
    def __init__(
        self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, omega_0=6, tmin=None, tmax=None, **kwargs
    ):
        super(SIREN, self).__init__(tmin=tmin, tmax=tmax, output_dim=output_dim, **kwargs)
        self.name = "SIREN"
        self.omega_0 = omega_0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_orig = x.clone()
        for layer in self.layers:
            x = torch.sin(self.omega_0 * layer(x))
        x = self.final_layer(x)
        return self.model_output_to_pulse_gradient(x, x_orig)


class RBFN(PulseGradientBase):
    def __init__(self, num_centers=10, center_spacing=1, output_dim=3, tmin=0, tmax=1, **kwargs):
        super(RBFN, self).__init__(tmin=tmin, tmax=tmax, output_dim=output_dim, **kwargs)
        self.centers = nn.Parameter(center_spacing * torch.randn(num_centers, 1))
        self.linear = nn.Linear(num_centers, output_dim)
        self.name = "RBFN"

    def forward(self, x):
        rbf = torch.exp(-torch.cdist(x, self.centers) ** 2)
        rbf = self.linear(rbf)
        return self.model_output_to_pulse_gradient(rbf, x)


class FourierMLP(PulseGradientBase):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=3,
        num_layers=3,
        num_fourier_features=10,
        frequency_scale=10,
        tmin=None,
        tmax=None,
        **kwargs,
    ):
        super(FourierMLP, self).__init__(tmin=tmin, tmax=tmax, output_dim=output_dim, **kwargs)
        self.fourier_weights = nn.Parameter(frequency_scale * torch.randn(num_fourier_features, input_dim))
        layers = [nn.Linear(num_fourier_features * 2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)
        self.name = "FourierMLP"

    def forward(self, x):
        out = torch.matmul(x, self.fourier_weights.T)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        out = self.model(out)
        return self.model_output_to_pulse_gradient(out, x)
