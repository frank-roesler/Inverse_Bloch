from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values


path = "/Users/frankrosler/Library/CloudStorage/Dropbox/160525_Mixed_smooth_1Slice/train_log.pt"

# B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
# G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
B1, G, target_z, target_xy, fixed_inputs = load_data(path)
target_z, target_xy, slice_centers, half_width = get_smooth_targets(theta=flip_angle, smoothness=3.0, function=torch.sigmoid, n_targets=n_slices)
# (B1, G, target_z, target_xy, pos, dt) = load_data_old(path)

shift = 0.7
exponent = 1j * torch.cumsum(G, dim=0) * fixed_inputs["dt"] * 1e3 * 2 * torch.pi * shift
B1_left = B1 * torch.exp(-exponent)
B1_right = B1 * torch.exp(exponent)
B1 = B1_left + B1_right

npts_off_resonance = 512
npts_some_b0_values = 3

freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
with torch.no_grad():
    plot_off_resonance(B1 + 0j, G, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz, path=path)
    plot_some_b0_values(npts_some_b0_values, fixed_inputs, G, B1 + 0j, target_xy, target_z, slice_centers, half_width, path=path)
    plot_timeprof(fixed_inputs, B1, G, fixed_inputs, slice_centers, path=path)
