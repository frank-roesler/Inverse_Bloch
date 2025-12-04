import torch
import torch.nn as nn


class FourierPulse(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, **kwargs):
        super().__init__()
        p = 1e-3 * torch.randn((2 * n_coeffs + 1, 2))
        weights = torch.exp(-0.01 * torch.arange(-n_coeffs, n_coeffs + 1) ** 2)
        p = p * weights.unsqueeze(1)
        self.params = torch.nn.Parameter(p)
        k = torch.arange(-n_coeffs, n_coeffs + 1, requires_grad=False).unsqueeze(0)
        self.freqs = torch.pi * k / (t_max - t_min)

    def to(self, device):
        self.k = self.k.to(device)
        return super().to(device)

    def forward(self, x):
        fx = self.freqs * x
        y_sin = torch.sum(self.params[:, 0] * torch.sin(fx), dim=-1, keepdim=True)
        y_cos = torch.sum(self.params[:, 1] * torch.cos(fx), dim=-1, keepdim=True)
        return (y_sin + y_cos).squeeze()


class FourierPulseOpt(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, **kwargs):
        super().__init__()
        p = 1e-3 * torch.randn((2 * n_coeffs + 1, 2))
        weights = torch.exp(-0.01 * torch.arange(-n_coeffs, n_coeffs + 1) ** 2)
        p = p * weights.unsqueeze(1)
        self.params = torch.nn.Parameter(p)
        k = torch.arange(-n_coeffs, n_coeffs + 1, requires_grad=False).unsqueeze(0)
        self.freqs = torch.pi * k / (t_max - t_min)

    def to(self, device):
        self.k = self.k.to(device)
        return super().to(device)

    def forward(self, x):
        fx = self.freqs * x  # (batch, k)
        sin_fx = torch.sin(fx)
        cos_fx = torch.cos(fx)
        # Use matmul for faster computation
        y_sin = sin_fx @ self.params[:, 0]
        y_cos = cos_fx @ self.params[:, 1]
        return y_sin + y_cos


class FourierSeries(nn.Module):
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
        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    # with torch.no_grad():
    t = torch.linspace(0, 1, 1000).unsqueeze(1)  # Shape (100, 1)
    torch.manual_seed(1)
    fs = FourierPulse(0, 1, n_coeffs=21)
    torch.manual_seed(1)
    fs_opt = FourierPulseOpt(0, 1, n_coeffs=21, output_dim=2)

    N_samples = 10000
    t0 = time()
    for _ in range(N_samples):
        y = fs(t).mean()
        y.backward()
    print(f"Time for {N_samples} forward passes (FourierPulse):", time() - t0)
    t0 = time()
    for _ in range(N_samples):
        y_opt = fs_opt(t).mean()
        y_opt.backward()
    print(f"Time for {N_samples} forward passes (FourierPulseOpt):", time() - t0)

    with torch.no_grad():
        plt.plot(t.numpy(), fs(t).detach().numpy(), label="y")
        plt.figure()
        plt.plot(t.numpy(), fs_opt(t).detach().numpy(), label="y_opt")
        plt.legend()
        plt.show()
