from utils_bloch.blochsim_CK import blochsim_CK
from utils_train.utils import *
import numpy as np
from params import *
import matplotlib.pyplot as plt

path = "results/train_log.pt"

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
target_z, target_xy, _, _ = get_test_targets()

# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
B1, G = load_data(path)

inputs["dt"] *= 2
dt = inputs["dt"]
t_B1 = torch.arange(0, len(B1)) * dt * 1e3
B1 /= 2
G /= 2

mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 1].plot(t_B1, G, linewidth=0.8)
ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
ax[1, 0].plot(fAx, np.real(mz.detach().cpu().numpy()), linewidth=0.8)
ax[1, 0].plot(fAx, target_z, linewidth=0.8)
ax[1, 1].plot(fAx, np.abs(mxy.detach().cpu().numpy()), linewidth=0.8)
ax[1, 1].plot(fAx, target_xy, linewidth=0.8)
plt.savefig("forward.png", dpi=300)
plt.show()
