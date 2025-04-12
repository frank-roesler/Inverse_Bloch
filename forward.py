from blochsim_CK import blochsim_CK
import numpy as np
from params import *
import matplotlib.pyplot as plt
import torch
from time import time

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch)

B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
G = torch.from_numpy(inputs["Gs"]).to(torch.float32)

t0 = time()
for i in range(10):
    mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)
print(time() - t0)

target_z = mz > 0.5
target_xy = np.abs(mxy) > 0.5

tAx = np.linspace(0, (len(B1) - 1) * dt, Nz)
bw = (1.0 / max(tAx)) * len(tAx)
fAx = np.linspace(-bw, bw, len(tAx)) * 1e-6 * 2 * np.pi
t_B1 = np.arange(0, len(B1)) * dt * 1e3


fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 1].plot(t_B1, inputs["Gs"], linewidth=0.8)
ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
ax[1, 0].plot(fAx, np.real(mz.detach().cpu().numpy()), linewidth=0.8)
# ax[1, 0].plot(fAx, target_z, linewidth=0.8)
ax[1, 1].plot(fAx, np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
# ax[1, 1].plot(fAx, target_xy, linewidth=0.8)
plt.savefig("forward.png", dpi=300)
plt.show()
