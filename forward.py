from utils_bloch.setup import get_targets
from utils_train.utils import *
import numpy as np
from params import *
import matplotlib.pyplot as plt
from utils_bloch.blochsim_batch import blochsim_CK_batch
from utils_bloch.blochsim_CK_freqprof import plot_off_resonance
from time import time

path = "results/280425_Mixed_square_flipAngle45/train_log.pt"

target_z, target_xy = get_targets(flip_angle)

# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
B1, G, axes = load_data(path)
t_B1 = axes["t_B1"]

npts = 512
gam = 267522.1199722082
gam_hz_mt = gam / (2 * np.pi)
freq_offsets_Hz = torch.linspace(-8000, 8000, npts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
freq_offsets_Hz, G, B1, B0, M0, sens, t_B1, pos, target_z, target_xy = move_to(
    (freq_offsets_Hz, G, B1, B0, M0, sens, t_B1, pos, target_z, target_xy), device
)
with torch.no_grad():
    plot_off_resonance(B1, G, pos, sens, dt, B0=B0, M0=M0, freq_offsets_Hz=freq_offsets_Hz)


freq_offsets_Hz = torch.linspace(-297.3 * 4.7 / gam_hz_mt, 0.0, 6)
B0_freq_offsets_mT = freq_offsets_Hz
B0_list = []
for ff in range(len(freq_offsets_Hz)):
    B0_list.append(B0 + B0_freq_offsets_mT[ff])

B0 = torch.stack(B0_list, dim=0).to(torch.float32)
mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=sens, B0_list=B0, M0=M0, dt=dt)
mxy = mxy.cpu().numpy()
mz = mz.cpu().numpy()
pos = pos.cpu().numpy()
target_xy = target_xy.cpu().numpy()
target_z = target_z.cpu().numpy()
t_B1 = t_B1.cpu().numpy()
G = G.cpu().numpy()
B1 = B1.cpu().numpy()
delta_t = np.diff(t_B1, axis=0)
for ff in range(npts):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
    ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
    ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
    ax[0, 1].plot(t_B1, G, linewidth=0.8)
    ax01 = ax[0, 1].twinx()
    ax01.plot([], [])
    ax01.plot(t_B1[:-1], np.diff(G, axis=0) / delta_t, linewidth=1)
    ax[1, 0].plot(pos[:, 2], np.real(mz[ff, :]), linewidth=0.8)
    ax[1, 0].plot(pos[:, 2], target_z, linewidth=0.8)
    ax[1, 1].plot(pos[:, 2], np.abs(mxy[ff, :]), linewidth=0.8)
    ax[1, 1].plot(pos[:, 2], target_xy, linewidth=0.8)
    # plt.savefig("forward.png", dpi=300)
    plt.show()
