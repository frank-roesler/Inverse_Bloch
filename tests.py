import torch
import matplotlib.pyplot as plt

# from utils_train.nets import *

# from utils_bloch.blochsim_CK import blochsim_CK
# from params import *

# from utils_train.utils import *


# print("MPS available:", torch.backends.mps.is_available())
# print("MPS built:", torch.backends.mps.is_built())
# device = (
#     torch.device("cpu")
#     if torch.backends.mps.is_available()
#     else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# )
# inputs, dt, Nz, sens, B0, G3, tAx, fAx, t_B1 = get_fixed_inputs(module=torch)
# tMin = t_B1[0]
# tMax = t_B1[-1]

# model = MLPWithBoundary(output_dim=3, left_boundary=tMin, right_boundary=tMax)
# model, optimizer_pulse, losses, best_loss, saved, save_timer = init_training(model, lr, device=device)

# pulse_gradient = model(t_B1.to(device))
# pulse = pulse_gradient[:, 0] + 1j * pulse_gradient[:, 1]
# gradient = gradient_scale * pulse_gradient[:, 2]

# print(pulse.shape)
# print(gradient.shape)
# print(inputs["rfmb"].shape)
# print(inputs["Gs"].shape)


# class FourierSeries(nn.Module):
#     # NEEDS TO BE UPDATED !!!
#     def __init__(self, n_coeffs=101, output_dim=1, tmin=0, tmax=1):
#         super().__init__()
#         tpulse = tmax - tmin
#         p = 1e-2 * torch.randn((n_coeffs, output_dim))
#         weights = torch.exp(-0.1 * torch.abs(torch.arange(0, n_coeffs)))
#         p = p * weights.unsqueeze(1)  # make Fourier coefficients decay to increase smoothness
#         self.params = torch.nn.Parameter(p)
#         self.k = (
#             torch.pi
#             * torch.arange(n_coeffs, requires_grad=False).unsqueeze(-1).repeat(1, 1, self.params.shape[1])
#             / tpulse
#         )

#     def forward(self, x):
#         kx = x.unsqueeze(-1) * self.k
#         out = torch.sum(self.params * torch.sin(kx), dim=1, keepdim=True).squeeze()
#         # y_cos = torch.sum(self.params[:, 1] * torch.cos(torch.pi * self.k * x / self.tpulse), dim=-1, keepdim=True)
#         return out  # + y_cos

#     def to(self, device):
#         self.k = self.k.to(device)
#         return super().to(device)


# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from params import *
# from utils import *
# from blochsim_CK import blochsim_CK

# torch.autograd.set_detect_anomaly(True)
# inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch)
# sens = sens.detach().requires_grad_(False)
# B0 = B0.detach().requires_grad_(False)
# tAx = tAx.detach().requires_grad_(False)
# fAx = fAx.detach().requires_grad_(False)
# t_B1 = t_B1.detach()
# _, _, target_z, target_xy = get_targets()
# targets_z = target_z.detach().requires_grad_(False)
# target_xy = target_xy.detach().requires_grad_(False)
# model = FourierSeries(n_coeffs=101, output_dim=3, tmin=t_B1[0].item(), tmax=t_B1[-1].item()).float()
# model, optimizer, losses = init_training(model, 1e-4)
# pulse_gradient = model(t_B1)
# pulse = pulse_gradient[:, 0:1] + 1j * pulse_gradient[:, 1:2]
# gradient = gradient_scale * pulse_gradient[:, 2:]
# mxy, mz = blochsim_CK(B1=pulse, G=gradient, sens=sens, B0=B0, **inputs)
# xy_profile_abs = torch.abs(mxy)
# loss = torch.mean((mz - target_z) ** 2) + torch.mean((xy_profile_abs - target_xy) ** 2)
# print("Loss", loss)
# losses.append(loss.item())
# optimizer.zero_grad()

# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}: {param.grad}")

# loss.backward()
# optimizer.step()

# print("-" * 100)
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}: {param.grad}")


dic = torch.load("results/FourierSeries_pre-trained_square/train_log.pt", weights_only=False)
pulse = dic["pulse"].detach().cpu()
grad = dic["gradient"].detach().cpu()

plt.plot(pulse)
plt.plot(torch.imag(pulse))
plt.figure()
plt.plot(grad)
plt.show()
