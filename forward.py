from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values


path = "/Users/frankrosler/Desktop/PhD/Python/Inverse_Bloch/results/160525_Mixed_smooth_1Slice/train_log.pt"

# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
(B1, G, target_z, target_xy, pos, dt, dx, sens, B0, tAx, t_B1, M0) = load_data(path)
target_z, target_xy, slice_centers, half_width = get_smooth_targets(theta=flip_angle, smoothness=3.0, function=torch.sigmoid, n_targets=n_slices)
# (B1, G, target_z, target_xy, pos, dt) = load_data_old(path)

shift = 0.7
B1_left = B1 * torch.exp(-1j * torch.cumsum(G, dim=0) * dt * 1e3 * 2 * torch.pi * shift)
B1_right = B1 * torch.exp(1j * torch.cumsum(G, dim=0) * dt * 1e3 * 2 * torch.pi * shift)
B1 = B1_left + B1_right


npts_off_resonance = 128
npts_some_b0_values = 11

device = get_device()
freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
freq_offsets_Hz, G, B1, B0, M0, sens, t_B1, pos, target_z, target_xy = move_to((freq_offsets_Hz, G, B1, B0, M0, sens, t_B1, pos, target_z, target_xy), device)
with torch.no_grad():
    plot_off_resonance(B1 + 0j, G, pos, sens, dt, B0=B0, M0=M0, freq_offsets_Hz=freq_offsets_Hz, path=path)
    plot_some_b0_values(npts_some_b0_values, pos, sens, G, B1 + 0j, B0, M0, target_xy, target_z, t_B1, dt, slice_centers, half_width, path=path)
    plot_timeprof(gamma, B1, G, pos, sens, B0_list, M0, dt, t_B1, slice_centers, path=path)
