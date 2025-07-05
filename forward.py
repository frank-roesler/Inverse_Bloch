from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values, plot_phase_fit_error, export_param_csv


def forward(path, npts_some_b0_values=7, Nz=4096, Nt=512, npts_off_resonance=512):
    (model, B1, G, target_z, target_xy, slice_centers_allB0, half_width, fixed_inputs) = load_data(path, mode="inference")

    n_b0_values = len(slice_centers_allB0)
    fixed_inputs = get_fixed_inputs(tfactor=tfactor, n_b0_values=n_b0_values, Nz=Nz, Nt=Nt, pos_spacing="linear")

    target_z, target_xy, slice_centers_allB0, half_width = get_smooth_targets(
        theta=flip_angle,
        smoothness=target_smoothness,
        function=torch.sigmoid,
        n_targets=n_slices,
        pos=fixed_inputs["pos"],
        n_b0_values=n_b0_values,
        shift_targets=shift_targets,
    )
    B1, G = model(fixed_inputs["t_B1"])

    print("PULS AMPLITUDE:", torch.max(torch.abs(B1)).item())

    freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
    with torch.no_grad():
        # export_param_csv(path, path)
        # plot_some_b0_values(npts_some_b0_values, fixed_inputs, G, B1, flip_angle, target_smoothness, n_slices, shift_targets, path=path)
        plot_timeprof(B1, G, fixed_inputs, slice_centers_allB0, half_width, path=path)
        # plot_phase_fit_error(fixed_inputs, B1, G, slice_centers_allB0, half_width, path=path)
        # plot_off_resonance(B1 + 0j, G, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz, flip_angle=flip_angle, path=path)
    plt.show()


if __name__ == "__main__":
    path = "results/Pulse_Comparison_0725/300625_Mixed_2Slice_phaseOffset200/train_log.pt"
    forward(path, npts_some_b0_values=7, Nz=1024, Nt=256, npts_off_resonance=512)
