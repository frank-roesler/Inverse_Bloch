from utils_bloch.blochsim_CK import blochsim_CK
from utils_train.utils import *
import numpy as np
from params import *
import matplotlib.pyplot as plt

path = "results/150425_MLP_square2/train_log.pt"

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
target_z, target_xy, _, _ = get_test_targets()

# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
B1, G = load_data(path)

# inputs["dt"] *= 2
# t_B1 = torch.arange(0, len(B1)) * inputs["dt"] * 1e3
# tAx = torch.linspace(0, (len(inputs["rfmb"]) - 1) * inputs["dt"], Nz)
# bw = (1.0 / max(tAx)) * len(tAx)
# fAx = torch.linspace(-bw, bw, len(tAx)) * 1e-6 * 2 * torch.pi
# B1 /= 2
# G /= 2

mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
ax[0, 1].plot(t_B1, G, linewidth=0.8)
ax[1, 0].plot(inputs["pos"][:, 2], np.real(mz.detach().cpu().numpy()), linewidth=0.8)
ax[1, 0].plot(inputs["pos"][:, 2], target_z, linewidth=0.8)
ax[1, 1].plot(inputs["pos"][:, 2], np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
ax[1, 1].plot(inputs["pos"][:, 2], target_xy, linewidth=0.8)
plt.savefig("forward.png", dpi=300)
plt.show()
