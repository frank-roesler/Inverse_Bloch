from utils_bloch.blochsim_CK import blochsim_CK
from utils_train.utils import *
import numpy as np
from params import *
import matplotlib.pyplot as plt
from utils_bloch.blochsim_CK_freqprof import plot_off_resonance
from time import time

path = "results/260425_Mixed_randomFitBs=10/train_log.pt"

target_z, target_xy = get_targets()

B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
# B1, G = load_data(path)

plot_off_resonance(B1.numpy(), G.numpy(), pos, Nz, dt)

mxy, mz = blochsim_CK(B1=B1, G=G, pos=pos, sens=sens, B0=B0, M0=M0, dt=dt)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
ax[0, 1].plot(t_B1, G, linewidth=0.8)
ax[1, 0].plot(pos[:, 2], np.real(mz.detach().cpu().numpy()), linewidth=0.8)
ax[1, 0].plot(pos[:, 2], target_z, linewidth=0.8)
ax[1, 1].plot(pos[:, 2], np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
ax[1, 1].plot(pos[:, 2], target_xy, linewidth=0.8)
plt.savefig("forward.png", dpi=300)
plt.show()
