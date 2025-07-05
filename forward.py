from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values, plot_fit_error, export_param_csv

path = "results/2025-07-04_15-38/train_log.pt"


(model, B1, G, target_z, target_xy, slice_centers_allB0, half_width, fixed_inputs) = load_data(path, mode="inference")

tfactor_new = 2
tfactor_old = 2
fixed_inputs = get_fixed_inputs(tfactor=tfactor_new, n_b0_values=n_b0_values, Nz=Nz, Nt=Nt, pos_spacing="linear")
B1, G = model(fixed_inputs["t_B1"] * tfactor_old / tfactor_new)
B1 = B1 * tfactor_old / tfactor_new
G = G * tfactor_old / tfactor_new


print("PULS AMPLITUDE:", torch.max(torch.abs(B1)).item())
npts_off_resonance = 256
npts_some_b0_values = 7

freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
with torch.no_grad():
    export_param_csv(path, path)
    plot_some_b0_values(npts_some_b0_values, fixed_inputs, G, B1, path=path)
    plot_timeprof(fixed_inputs, B1, G, fixed_inputs, slice_centers_allB0, path=path, fig=None, ax=None)
    plot_fit_error(fixed_inputs, B1, G, slice_centers_allB0, half_width, path=path)
    plot_off_resonance(B1 + 0j, G, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz, path=path, block=True)
plt.show()
