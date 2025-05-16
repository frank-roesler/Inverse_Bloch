import numpy as np
from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer.plot_off_resonance import *

path = "results/train_log.pt"

# target_z, target_xy = get_smooth_targets(theta=flip_angle, smoothness=3.0, function=torch.sigmoid, n_targets=n_slices)
# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
(
    B1,
    G,
    target_z,
    target_xy,
    pos,
    dt,
    dx,
    sens,
    B0,
    tAx,
    t_B1,
    M0,
) = load_data(path)


npts = 128
freq_offsets_Hz = torch.linspace(-8000, 8000, npts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
freq_offsets_Hz, G, B1, B0, M0, sens, t_B1, pos, target_z, target_xy = move_to((freq_offsets_Hz, G, B1, B0, M0, sens, t_B1, pos, target_z, target_xy), device)
with torch.no_grad():
    plot_off_resonance(B1, G, pos, sens, dt, B0=B0, M0=M0, freq_offsets_Hz=freq_offsets_Hz)
    # plot_some_b0_values(6, pos, sens, G, B1, B0, M0, target_xy, target_z, t_B1, dt)
