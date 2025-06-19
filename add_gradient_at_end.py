import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_bloch import blochsim_CK_batch
from utils_bloch.setup import *
from utils_train.utils import *
from params import *

path = "/Users/frankrosler/Library/CloudStorage/Dropbox/120625_Mixed_90deg_phaseDiffLoss/train_log.pt"


def simulate_B0_values(fixed_inputs, B1, G, M0=fixed_inputs["M0"]):
    with torch.no_grad():
        gamma_hz_mt = fixed_inputs["gam_hz_mt"]
        pos = fixed_inputs["pos"]
        freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, 2)
        freq_offsets_Hz = torch.mean(freq_offsets_Hz, dim=0)
        B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
        B0_list = [fixed_inputs["B0"] + B0_freq_offsets_mT]

        B0 = torch.stack(B0_list, dim=0).to(torch.float32)
        mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=fixed_inputs["sens"], B0_list=B0, M0=M0, dt=fixed_inputs["dt_num"])
    return mxy, mz


B1, G, target_z, target_xy, fixed_inputs = load_data(path)

shift = 0.0025
exponent = 1j * torch.cumsum(G, dim=0) * fixed_inputs["dt_num"] * 2 * torch.pi * shift * fixed_inputs["gam"]
B1_left = B1 * torch.exp(-exponent)
B1_right = B1 * torch.exp(exponent)
B1 = B1_left + B1_right
n_slices = 2
target_z, target_xy, slice_centers, half_width = get_smooth_targets(theta=flip_angle, smoothness=3.0, function=torch.sigmoid, n_targets=n_slices)

mxy, mz = simulate_B0_values(fixed_inputs, B1, G)
phase = np.unwrap(np.angle(mxy[0, :]))


fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

# First plot
ax2 = ax1.twinx()
ax1.plot(fixed_inputs["pos"], np.abs(mxy[0, :]), label="|mxy|")
ax1.plot(fixed_inputs["pos"], mz[0, :], label="mz")
ax2.plot(fixed_inputs["pos"], phase, "g--", label="phase", linewidth=0.8)
ax1.set_ylabel("Magnitude")
ax2.set_ylabel("Phase (rad)", color="g")
ax1.set_xlim(-0.05, 0.05)
ax2.set_ylim(-200, -120)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("Original")

M0 = torch.zeros((3, mxy.shape[-1]), dtype=torch.float32)
M0[0, :] = mxy.real
M0[1, :] = mxy.imag
M0[2, :] = mz
mxy_new, mz_new = simulate_B0_values(fixed_inputs, B1 * 0, -0.45 * G, M0=M0)
phase = np.unwrap(np.angle(mxy_new[0, :]))
slices = np.abs(mxy_new[0, :]) > 0.05
phasemin = np.min(phase[slices])
phasemax = np.max(phase[slices])

# Second plot
ax4 = ax3.twinx()
ax3.plot(fixed_inputs["pos"], np.abs(mxy_new[0, :]), label="|mxy_new|")
ax3.plot(fixed_inputs["pos"], mz_new[0, :], label="mz_new")
ax4.plot(fixed_inputs["pos"], phase, "g--", label="phase_new", linewidth=0.8)
ax3.set_xlabel("Position (z)")
ax3.set_ylabel("Magnitude")
ax4.set_ylabel("Phase (rad)", color="g")
ax3.set_xlim(-0.05, 0.05)
ax4.set_ylim(phasemin, phasemax)
ax3.legend(loc="upper left")
ax4.legend(loc="upper right")
ax3.set_title("After Gradient")

fig.tight_layout()
plt.show()
