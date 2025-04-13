from utils_bloch.blochsim_CK import blochsim_CK
import numpy as np
from params import *
import matplotlib.pyplot as plt
import torch
from time import time

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
target_z, target_xy, _, _ = get_test_targets()

B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
G = torch.from_numpy(inputs["Gs"]).to(torch.float32)

mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 1].plot(t_B1, inputs["Gs"], linewidth=0.8)
ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
ax[1, 0].plot(fAx, np.real(mz.detach().cpu().numpy()), linewidth=0.8)
ax[1, 0].plot(fAx, target_z, linewidth=0.8)
ax[1, 1].plot(fAx, np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
ax[1, 1].plot(fAx, target_xy, linewidth=0.8)
plt.savefig("forward.png", dpi=300)
plt.show()
