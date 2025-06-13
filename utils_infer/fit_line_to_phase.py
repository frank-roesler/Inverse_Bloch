import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_bloch import blochsim_CK_batch


def simulate_B0_values(fixed_inputs, B1, G, n_b0_values=3):
    gamma_hz_mt = fixed_inputs["gam_hz_mt"]
    pos = fixed_inputs["pos"]
    freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, n_b0_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(fixed_inputs["B0"] + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=fixed_inputs["sens"], B0_list=B0, M0=fixed_inputs["M0"], dt=fixed_inputs["dt"])
    return mxy, mz


def fit_line_to_phase(fixed_inputs, B1, G, centers, half_width):
    mxy, mz = simulate_B0_values(fixed_inputs, B1, G, n_b0_values=3)
    fitted_lines = []
    phases = []
    for ff in range(mxy.shape[0]):
        phase = np.unwrap(np.angle(mxy[ff, :]))
        n_slices = len(centers)
        fitted_line = np.zeros_like(phase)

        for i in range(n_slices):
            center = centers[i]
            start = max(center - half_width, 0)
            end = min(center + half_width, len(phase) - 1)

            x = np.arange(start, end + 1, dtype=np.float32)
            y = phase[start : end + 1]

            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                fitted_line[start : end + 1] = slope * x + intercept
        fitted_lines.append(fitted_line)
        phases.append(phase)

    return fitted_lines, phases


def plot_fit_error(fixed_inputs, B1, G, centers, half_width):
    fitted_lines, phases = fit_line_to_phase(fixed_inputs, B1, G, centers, half_width)
    where_slices_are = np.zeros_like(phases[0])
    for center in centers:
        where_slices_are[center - half_width : center + half_width] = 1
    plt.figure(figsize=(10, 5))
    for i, fitted_line in enumerate(fitted_lines):
        error = phases[i] - fitted_line
        error[~where_slices_are.astype(bool)] = np.nan
        plt.plot(fixed_inputs["pos"][:, 2], error, linewidth=0.8)
    plt.title("Phase Fitting and Error")
    plt.xlabel("pos")
    plt.ylabel("Phase Value")
