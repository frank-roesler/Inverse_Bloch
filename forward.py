from utils_bloch.setup import *
from utils_train.utils import *
from params import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values, plot_phase_fit_error, export_param_csv


def forward(path, npts_some_b0_values=7, Nz=4096, Nt=512, npts_off_resonance=512):
    (model, B1, G, _, _, slice_centers_allB0, half_width, shift_targets, n_slices, fixed_inputs) = load_data(path, mode="inference")

    n_b0_values = len(slice_centers_allB0)
    fixed_inputs = get_fixed_inputs(tfactor=tfactor, n_b0_values=n_b0_values, Nz=Nz, Nt=Nt, pos_spacing="linear")
    B1, G = model(fixed_inputs["t_B1"])

    # shift = 0.0025
    # exponent = 1j * torch.cumsum(G, dim=0) * fixed_inputs["dt_num"] * 2 * torch.pi * shift * fixed_inputs["gam"]
    # B1_left = B1 * torch.exp(-exponent)
    # B1_right = B1 * torch.exp(exponent)
    # B1_lleft = B1_left * torch.exp(-2 * exponent)
    # B1_rright = B1_right * torch.exp(2 * exponent)
    # B1 = B1_left + B1_right + B1_lleft + B1_rright
    # n_slices = 4

    _, _, slice_centers_allB0, half_width = get_smooth_targets(
        theta=flip_angle,
        smoothness=target_smoothness,
        function=torch.sigmoid,
        n_targets=n_slices,
        pos=fixed_inputs["pos"],
        n_b0_values=n_b0_values,
        shift_targets=shift_targets,
    )

    print("PULS AMPLITUDE:", torch.max(torch.abs(B1)).item())
    print("GRADIENT_MOMENT:", torch.sum(G, dim=0).item() * fixed_inputs["dt_num"])

    freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
    with torch.no_grad():
        # plot_some_b0_values(npts_some_b0_values, fixed_inputs, G, B1, flip_angle, target_smoothness, n_slices, shift_targets, path=path)
        # plot_timeprof(B1, G, fixed_inputs, slice_centers_allB0, half_width, path=path)
        slopes = plot_phase_fit_error(fixed_inputs, B1, G, slice_centers_allB0, half_width, count_slices_with_algorithm=False, path=path)
        export_param_csv(path, path, B1, G, fixed_inputs, slopes)
        plot_off_resonance(B1 + 0j, G, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz, flip_angle=flip_angle, path=path)
    # plt.show()


if __name__ == "__main__":
    path = "results/old/Pulse_Comparison_0725/300625_Mixed_2Slice_phaseOffset200/train_log.pt"
    forward(path, npts_some_b0_values=7, Nz=4096, Nt=128, npts_off_resonance=4096)
