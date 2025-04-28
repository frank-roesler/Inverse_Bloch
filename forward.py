from utils_bloch.blochsim_CK import blochsim_CK
from utils_train.utils import *
import numpy as np
from params import *
import matplotlib.pyplot as plt
from utils_bloch.blochsim_CK_freqprof import plot_off_resonance
from aux.blochsim_CK_freqprof_old import plot_off_resonance_old
from time import time

path = "results/260425_Mixed_randomFitBs=10/train_log.pt"

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
target_z, target_xy = get_test_targets()

B1, G, inputs, targets = load_data(path)
target_z = targets["target_z"]
target_xy = targets["target_xy"]
dt = inputs["dt"]
t_B1 = torch.arange(0, len(inputs["rfmb"])) * dt * 1e3
gam = 267522.1199722082
gam_hz_mt = gam / (2 * np.pi)
freq_offset = -297.3 * 4.7 / gam_hz_mt / 2

with torch.no_grad():
    plot_off_resonance(B1.numpy(), G.numpy(), inputs["pos"], Nz, dt)
    mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0 + freq_offset, **inputs)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
ax[0, 1].plot(t_B1, G, linewidth=0.8)
ax[1, 0].plot(inputs["pos"][:, 2], np.real(mz.detach().cpu().numpy()), linewidth=0.8)
ax[1, 0].plot(inputs["pos"][:, 2], target_z, linewidth=0.8)
ax[1, 1].plot(inputs["pos"][:, 2], np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
ax[1, 1].plot(inputs["pos"][:, 2], target_xy, linewidth=0.8)
ax[1, 0].set_xlim(-0.05, 0.05)
ax[1, 1].set_xlim(-0.05, 0.05)
ax_phase = ax[1, 1].twinx()
ax_phase.set_ylabel("Phase (radians)")
phase = np.unwrap(np.angle(mxy.detach().cpu().numpy()))
phasemin = np.min(phase)
phasemax = np.max(phase)
phase[target_xy < 0.5] = np.nan
ax_phase.plot(inputs["pos"][:, 2], phase, linewidth=0.8, color="g")
plt.savefig(f"forward.png", dpi=300)

# mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0 + 297.3 * 4.7, **inputs)

# fig2, ax2 = plt.subplots(2, 2, figsize=(12, 6))
# ax2[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
# ax2[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
# ax2[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
# ax2[0, 1].plot(t_B1, G, linewidth=0.8)
# ax2[1, 0].plot(inputs["pos"][:, 2], np.real(mz.detach().cpu().numpy()), linewidth=0.8)
# ax2[1, 0].plot(inputs["pos"][:, 2], target_z, linewidth=0.8)
# ax2[1, 1].plot(inputs["pos"][:, 2], np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
# ax2[1, 1].plot(inputs["pos"][:, 2], target_xy, linewidth=0.8)
# ax_phase2 = ax2[1, 1].twinx()
# ax_phase2.set_ylabel("Phase (radians)")
# phase = np.unwrap(np.angle(mxy.detach().cpu().numpy()))
# phasemin = np.min(phase)
# phasemax = np.max(phase)
# phase[target_xy < 0.5] = np.nan
# ax_phase2.plot(inputs["pos"][:, 2], phase, linewidth=0.8, color="g")
# plt.savefig("forward0.png", dpi=300)


plt.show()
