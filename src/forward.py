from utils_bloch.setup import *
from utils_train.utils import *
from config import *
from utils_infer import plot_timeprof, plot_off_resonance, plot_some_b0_values, plot_phase_fit_error, export_param_csv, export_as_numpy


def forward(path, npts_some_b0_values=7, Nz=512, Nt=512, npts_off_resonance=512):
    (model, pulse, gradient, _, target_xy, tconfig, bconfig) = load_data(path, mode="inference")

    n_b0_values = target_xy.shape[0]
    forward_inputs = get_fixed_inputs(tfactor=bconfig.tfactor, n_b0_values=n_b0_values, Nz=Nz, Nt=Nt, pos_spacing="linear")
    B1, G = model(forward_inputs["t_B1"])

    # t = torch.linspace(0, forward_inputs["t_B1"][-1].item(), 256)
    # B1, G = model(t.unsqueeze(1))
    # np.save("results/2025-12-25_22-35/gradient_numpy.npy", G.detach().numpy())
    # np.save("results/2025-12-25_22-35/pulse_numpy.npy", B1.detach().numpy())

    target_xy = get_smooth_targets(tconfig, bconfig, function=torch.sigmoid, override_inputs=forward_inputs)

    print("PULS AMPLITUDE:", torch.max(torch.abs(B1)).item())
    print("GRADIENT_MOMENT:", torch.sum(G, dim=0).item() * forward_inputs["dt_num"])

    freq_offsets_Hz = torch.linspace(-8000, 8000, npts_off_resonance)
    with torch.no_grad():
        plot_some_b0_values(npts_some_b0_values, forward_inputs, G, B1, tconfig, bconfig, path=path)
        plot_timeprof(B1, G, forward_inputs, target_xy, path=path)
        slope = plot_phase_fit_error(forward_inputs, target_xy, B1, G, path=path)
        export_param_csv(path, path, B1, G, forward_inputs, slope)
        plot_off_resonance(B1 + 0j, G, forward_inputs, freq_offsets_Hz=freq_offsets_Hz, flip_angle=bconfig.flip_angle, path=path)
    # plt.show()


if __name__ == "__main__":
    path = "results/2026-01-26_12-37/train_log.pt"
    forward(path, npts_some_b0_values=8, Nz=2048, Nt=256, npts_off_resonance=512)
    # torch_path = "results/2025-12-13_18-52/train_log.pt"
    # export_as_numpy(torch_path, "results/numpy_pulses")
