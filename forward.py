from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values, plot_fit_error


path = "results/train_log.pt"

target_z, target_xy, slice_centers, half_width = get_smooth_targets(theta=flip_angle, smoothness=3.0, function=torch.sigmoid, n_targets=n_slices, pos=fixed_inputs["pos"])
# _, _, _, _, B1, G, target_z, target_xy, fixed_inputs = load_data_old(path)
B1, G, target_z, target_xy, fixed_inputs = load_data(path)

fig, ax = plt.subplots(figsize=(12, 6))
for Nt in [64]:
    (model, target_z, target_xy, optimizer, losses, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, start_epoch) = load_data(path, mode="train")
    fixed_inputs = get_fixed_inputs(tfactor=4.0, n_b0_values=1, Nz=256, Nt=Nt)
    B1, G = model(fixed_inputs["t_B1"])

    shift = 0.0025
    exponent = 1j * torch.cumsum(G, dim=0) * fixed_inputs["dt_num"] * 2 * torch.pi * shift * fixed_inputs["gam"]
    B1_left = B1 * torch.exp(-exponent)
    B1_right = B1 * torch.exp(exponent)
    B1 = B1_left + B1_right
    n_slices = 2
    target_z, target_xy, slice_centers, half_width = get_smooth_targets(theta=flip_angle, smoothness=2.0, function=torch.sigmoid, n_targets=n_slices, pos=fixed_inputs["pos"])

    print("PULS AMPLITUDE:", torch.max(torch.abs(B1)).item())

    npts_off_resonance = 256
    npts_some_b0_values = 5

    freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
    with torch.no_grad():
        plot_off_resonance(B1 + 0j, G, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz, path=path, block=True)
        plot_some_b0_values(npts_some_b0_values, fixed_inputs, G, B1, target_xy, target_z, slice_centers, half_width, path=path)
        plot_timeprof(fixed_inputs, B1, G, fixed_inputs, slice_centers, path=path, fig=fig, ax=ax)
        # plot_fit_error(fixed_inputs, B1, G, slice_centers, half_width)
plt.show()
