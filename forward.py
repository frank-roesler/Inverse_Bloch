from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values, plot_fit_error, export_param_csv

path = "results/train_log.pt"

(model, B1, G, target_z, target_xy, slice_centers_allB0, half_width, fixed_inputs) = load_data(path, mode="inference")
# (model, target_z, target_xy, optimizer, losses, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, model_args, epoch) = load_data(path, mode="train")

tfactor_new = 2
tfactor_old = 2
fixed_inputs = get_fixed_inputs(tfactor=tfactor_new, n_b0_values=n_b0_values, Nz=Nz, Nt=Nt, pos_spacing="linear")
B1, G = model(fixed_inputs["t_B1"] * tfactor_old / tfactor_new)
B1 = B1 * tfactor_old / tfactor_new
G = G * tfactor_old / tfactor_new

# trainLogger = TrainLogger(
#     fixed_inputs,
#     flip_angle,
#     loss_metric,
#     {"target_z": target_z, "target_xy": target_xy},
#     model_args,
#     scanner_params,
#     loss_weights,
#     n_slices,
#     n_b0_values,
#     tfactor,
#     Nz,
#     Nt,
#     start_logging=0,
# )
# trainLogger.log_epoch(epoch, losses[-1] * torch.ones(1), losses, model, optimizer)

# shift = 0.0025
# exponent = 1j * torch.cumsum(G, dim=0) * fixed_inputs["dt_num"] * 2 * torch.pi * shift * fixed_inputs["gam"]
# B1_left = B1 * torch.exp(-exponent)
# B1_right = B1 * torch.exp(exponent)
# B1 = B1_left + B1_right
# n_slices = 2


print("PULS AMPLITUDE:", torch.max(torch.abs(B1)).item())
npts_off_resonance = 256
npts_some_b0_values = 7

freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
with torch.no_grad():
    export_param_csv(path, path)
    # plot_some_b0_values(npts_some_b0_values, fixed_inputs, G, B1, path=path)
    # plot_timeprof(fixed_inputs, B1, G, fixed_inputs, slice_centers_allB0, path=path, fig=None, ax=None)
    # plot_fit_error(fixed_inputs, B1, G, slice_centers_allB0, half_width, path=path)
    # plot_off_resonance(B1 + 0j, G, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz, path=path, block=True)
plt.show()
