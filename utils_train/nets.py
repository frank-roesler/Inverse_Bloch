import torch.nn as nn
import torch
import torch.nn.init as init
from math import sqrt
import torch.nn.functional as F


def get_model(model_name, **kwargs):
    """Get model by name."""
    model_dict = {
        "MLP": MLP,
        "SIREN": SIREN,
        "FourierSeries": FourierSeries,
        "RBFN": RBFN,
        "FourierMLP": FourierMLP,
        "MixedModel": MixedModel,
        "ModulatedFourier": ModulatedFourier,
        "MixedModel_RealPulse": MixedModel_RealPulse,
        "NoModel": NoModel,
    }
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} is not supported.")
    return model_dict[model_name](**kwargs).float()


class PulseGradientBase(nn.Module):
    def __init__(self, gradient_scale, positive_gradient=True, tmin=None, tmax=None, output_dim=3, **kwargs):
        super(PulseGradientBase, self).__init__()
        self.gradient_scale = sqrt(gradient_scale)
        self.positive_gradient = positive_gradient
        self.tmin = tmin
        self.tmax = tmax
        self.output_dim = output_dim
        self.name = "no_model_name"

    def model_output_to_pulse_gradient(self, model_output, x):
        if self.output_dim != 3:
            return model_output
        out_sign = nn.Softplus() if self.positive_gradient else nn.Identity()
        pulse = model_output[:, 0:1] + 1j * model_output[:, 1:2]
        gradient = self.gradient_scale * out_sign(self.gradient_scale * model_output[:, 2:])
        if self.tmin is None or self.tmax is None:
            return pulse, gradient
        bdry_scaling = (x - self.tmin) * (self.tmax - x)
        return pulse * bdry_scaling, gradient


class MixedModel(PulseGradientBase):
    def __init__(self, tmin=None, tmax=None, **kwargs):
        super(MixedModel, self).__init__(tmin=None, tmax=None, **kwargs)
        self.name = "MixedModel"
        self.model1 = FourierPulse(tmin, tmax, **kwargs)
        self.model2 = FourierPulse(tmin, tmax, **kwargs)
        self.model3 = SIREN(**kwargs, output_dim=1)
        # self.model3 = MLP(**kwargs, output_dim=1)

    def to(self, device):
        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)
        self.model3 = self.model3.to(device)
        return super().to(device)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out = torch.cat([out1, out2, out3], dim=-1)
        return self.model_output_to_pulse_gradient(out, x)


class FourierPulse(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, **kwargs):
        super().__init__()
        self.tpulse = t_max - t_min
        p = 1e-3 * torch.randn((2 * n_coeffs + 1, 2))
        # p[:, 1] *= 1e-6
        weights = torch.exp(-0.01 * torch.arange(-n_coeffs, n_coeffs + 1) ** 2)
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
        super().__init__(tmin=None, tmax=None, output_dim=output_dim, **kwargs)
        self.pulses = nn.ModuleList([FourierPulse(tmin, tmax, n_coeffs=n_coeffs) for _ in range(output_dim)])
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
                nn.init.uniform_(layer.bias, -0.001, 0.001)

    def forward(self, x):
        output = self.model(x)
        return self.model_output_to_pulse_gradient(output, x)


class SIREN(PulseGradientBase):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3, num_layers=3, omega_0=6, tmin=None, tmax=None, **kwargs):
        super(SIREN, self).__init__(tmin=tmin, tmax=tmax, output_dim=output_dim, **kwargs)
        self.name = "SIREN"
        self.omega_0 = omega_0 / num_layers
        self.layers = nn.ModuleList()
        initial_weight_bound = 0.1
        layer0 = nn.Linear(input_dim, hidden_dim)
        self.layers.append(layer0)
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        init.uniform_(self.final_layer.weight, a=-initial_weight_bound, b=initial_weight_bound)
        init.uniform_(self.final_layer.bias, a=-initial_weight_bound, b=initial_weight_bound)

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
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3, num_layers=3, num_fourier_features=10, frequency_scale=10, tmin=None, tmax=None, **kwargs):
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


class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class ModulatedFourier(PulseGradientBase):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3, bandwidth=30, tmin=None, tmax=None, **kwargs):
        super(ModulatedFourier, self).__init__(tmin=tmin, tmax=tmax, output_dim=output_dim, **kwargs)
        self.name = "ModulatedFourier"
        self.output_dim = output_dim
        linear = nn.Linear(input_dim, hidden_dim)
        linear2 = nn.Linear(hidden_dim, output_dim)
        init.uniform_(linear.weight, a=-bandwidth, b=bandwidth)
        init.uniform_(linear.bias, a=-1, b=1)
        init.uniform_(linear2.weight, a=-0.005, b=0.005)
        init.uniform_(linear2.bias, a=-0.005, b=0.005)
        self.layers1 = nn.Sequential(linear, SineActivation(), linear2)

    def forward(self, x):
        out = self.layers1(x)
        return self.model_output_to_pulse_gradient(out, x)


class MixedModel_RealPulse(PulseGradientBase):
    def __init__(self, tmin=None, tmax=None, **kwargs):
        super(MixedModel_RealPulse, self).__init__(tmin=None, tmax=None, **kwargs)
        self.name = "MixeMdodel_RealPulse"
        self.pulse_model = FourierPulse(tmin, tmax, **kwargs)
        self.gradient_value = torch.nn.Parameter(torch.randn(1))

    def to(self, device):
        self.pulse_model = self.pulse_model.to(device)
        # self.gradient_value = torch.nn.Parameter(self.gradient_value.to(device))
        return super().to(device)

    def forward(self, x):
        pulse = self.pulse_model(x)
        gradient = self.gradient_scale * self.gradient_value

        return pulse, gradient


class NoModel(PulseGradientBase):
    def __init__(self, input_dim=1, output_dim=3, tvector=torch.arange(0, 512) / 512 * 1.25, **kwargs):
        super(NoModel, self).__init__(output_dim=output_dim, **kwargs)
        self.name = "NoModel"
        self.Nt = len(tvector)
        p = 1e-3 * torch.randn(len(tvector), 3)

        p = self.smooth_params(p)
        p = self.smooth_params(p)
        p = self.smooth_params(p)
        p = self.smooth_params(p)

        bdry_scaling = (tvector - self.tmin) * (self.tmax - tvector) / (self.tmax - self.tmin) if self.tmin is not None and self.tmax is not None else torch.ones_like(tvector)
        p[:, 0:2] = p[:, 0:2] * bdry_scaling.unsqueeze(1)
        p[:, 2] = sqrt(self.gradient_scale) * (100 * sqrt(self.gradient_scale) * p[:, 2] + 1)
        self.params = torch.nn.Parameter(p)

    def smooth_params(self, p):
        smoothing_kernel_size = 10
        pad = smoothing_kernel_size // 2
        p_reshaped = p.T.unsqueeze(0)  # [1, 3, 512]
        p_pad = torch.nn.functional.pad(p_reshaped, (pad, pad), mode="reflect")
        p_smooth = torch.nn.functional.avg_pool1d(p_pad, kernel_size=smoothing_kernel_size, stride=1)
        p_smooth = p_smooth.squeeze(0).T  # [513, 3]
        p_smooth = p_smooth[: self.Nt]  # Slice to [512, 3]
        norm = torch.norm(p, dim=1, keepdim=True).clamp(min=1e-8)
        norm_smooth = torch.norm(p_smooth, dim=1, keepdim=True).clamp(min=1e-8)
        return p_smooth * (norm / norm_smooth)

    def forward(self, x):
        pulse = self.params[:, 0:1] + 1j * self.params[:, 1:2]
        gradient = self.params[:, 2:]
        return pulse, gradient
