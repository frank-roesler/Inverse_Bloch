from utils_bloch.blochsim_CK import blochsim_CK
from utils_train.utils import *
import numpy as np
from params import *
import matplotlib.pyplot as plt
from utils_bloch.blochsim_CK_freqprof import plot_off_resonance
from aux.blochsim_CK_freqprof_old import plot_off_resonance_old
from time import time

path = "results/210425_Mixed_square/train_log.pt"

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
target_z, target_xy, _, _ = get_test_targets()

# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
B1, G = load_data(path)

t0 = time()
plot_off_resonance(B1.numpy(), G.numpy(), inputs["pos"], Nz, dt)
dt1 = time() - t0
t1 = time()
plot_off_resonance_old(B1.numpy(), G.numpy(), inputs["pos"], Nz, dt)
dt2 = time() - t1
print("-" * 100)
print(f"New: {dt1:.2f} s")
print(f"Old: {dt2:.2f} s")

# mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)

# fig, ax = plt.subplots(2, 2, figsize=(12, 6))
# ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
# ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
# ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
# ax[0, 1].plot(t_B1, G, linewidth=0.8)
# ax[1, 0].plot(inputs["pos"][:, 2], np.real(mz.detach().cpu().numpy()), linewidth=0.8)
# ax[1, 0].plot(inputs["pos"][:, 2], target_z, linewidth=0.8)
# ax[1, 1].plot(inputs["pos"][:, 2], np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
# ax[1, 1].plot(inputs["pos"][:, 2], target_xy, linewidth=0.8)
# plt.savefig("forward.png", dpi=300)
# plt.show()
