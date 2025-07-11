import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_bloch import blochsim_CK_batch
import os


def simulate_B0_values(fixed_inputs, B1, G, n_b0_values=3):
    gamma_hz_mt = fixed_inputs["gam_hz_mt"]
    pos = fixed_inputs["pos"]
    freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, n_b0_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(fixed_inputs["B0"] + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=fixed_inputs["sens"], B0_list=B0, M0=fixed_inputs["M0"], dt=fixed_inputs["dt_num"])
    return mxy, mz


def fit_line_to_phase(fixed_inputs, B1, G, centers, half_width):
    n_slices = len(centers[0])
    n_b0_values = len(centers)
    mxy, mz = simulate_B0_values(fixed_inputs, B1, G, n_b0_values=n_b0_values)
    fitted_lines = []
    phases = []
    slopes = []
    for ff in range(mxy.shape[0]):
        phase = np.unwrap(np.angle(mxy[ff, :]))
        fitted_line = np.zeros_like(phase)
        allPhases = np.zeros((2 * half_width + 1))
        x = np.arange(-half_width, half_width + 1, dtype=np.float32) / (2 * half_width) * 0.02
        for i in range(n_slices):
            center = centers[ff][i]
            start = max(center - half_width, 0)
            end = min(center + half_width, len(phase) - 1)
            allPhases += phase[start : end + 1] - np.mean(phase[start : end + 1])

        slope, intercept = np.polyfit(x, allPhases / n_slices, 1)
        slopes.append(slope)
        for i in range(n_slices):
            center = centers[ff][i]
            start = max(center - half_width, 0)
            end = min(center + half_width, len(phase) - 1)
            fitted_line[start : end + 1] = slope * x + np.mean(phase[start : end + 1])
        fitted_lines.append(fitted_line)
        phases.append(phase)
    return fitted_lines, phases, slopes


def plot_phase_fit_error(fixed_inputs, B1, G, centers_allB0, half_width, path=None):
    fitted_lines, phases, slopes = fit_line_to_phase(fixed_inputs, B1, G, centers_allB0, half_width)
    where_slices_are = np.zeros((len(centers_allB0), phases[0].shape[0]))
    for i, centers in enumerate(centers_allB0):
        for center in centers:
            where_slices_are[i, center - half_width : center + half_width] = 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fitted_line in enumerate(fitted_lines):
        error = phases[i] - fitted_line
        error[~where_slices_are[i].astype(bool)] = np.nan
        plt.plot(fixed_inputs["pos"], error, linewidth=0.8, label=f"{i}")
    plt.title("Phase Fitting and Error")
    plt.xlabel("pos")
    plt.ylabel("Phase Value")
    plt.legend()

    if path is not None:
        fig.savefig(os.path.join(os.path.dirname(path), f"phase_error.png"), dpi=300)

    return slopes
