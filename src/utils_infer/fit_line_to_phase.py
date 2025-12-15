import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_bloch import blochsim_CK_batch
import os
import matplotlib.cm as cm
from scipy.signal import argrelextrema
from constants import gamma, gam_hz_mt, larmor_mhz, water_ppm


def simulate_B0_values(fixed_inputs, B1, G, n_b0_values=3):
    freq_offsets_Hz = torch.linspace(-larmor_mhz * water_ppm, 0.0, n_b0_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(fixed_inputs["B0"] + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=fixed_inputs["pos"], sens=fixed_inputs["sens"], B0_list=B0, M0=fixed_inputs["M0"], dt=fixed_inputs["dt_num"])
    return mxy, mz


def count_slices(abs_mxy):
    """Attempts to count the actual slices from the profile m_xy by computing intersection points with a constant line at 0.5.
    If the number of intersection points is not even, the procedure is deemed unsuccessful."""
    abs_mxy = np.clip(abs_mxy, 0.4, 0.6).numpy()
    argmins = argrelextrema(np.abs(abs_mxy - 0.5), np.less)[0]
    if not len(argmins) % 2 == 0:
        return (([], []), False)
    centers = [(argmins[i].item() + argmins[i + 1].item()) // 2 for i in 2 * np.arange(len(argmins) // 2)]
    half_widths = [(argmins[i + 1] - argmins[i]) / 2 for i in 2 * np.arange(len(argmins) // 2)]
    half_width = int(np.mean(half_widths).item())
    return ((centers, half_width), True)


def fit_line_to_phase2(fixed_inputs, B1, G, centers, half_width):
    n_slices = len(centers[0])
    n_b0_values = len(centers)
    mxy, mz = simulate_B0_values(fixed_inputs, B1, G, n_b0_values=n_b0_values)
    phases = []
    slopes = []
    x = np.arange(-half_width, half_width + 1, dtype=np.float32) / (2 * half_width) * 0.02
    for ff in range(mxy.shape[0]):
        phase = np.unwrap(np.angle(mxy[ff, :]))
        for i in range(n_slices):
            center = centers[ff][i]
            start = max(center - half_width, 0)
            end = min(center + half_width, len(phase) - 1)
            current_phase = phase[start : end + 1] - np.mean(phase[start : end + 1])
            current_slope, _ = np.polyfit(x, current_phase, 1)
            phases.append(current_phase)
            slopes.append(current_slope)
    mean_slope = np.mean(slopes)
    remaining_variation = float("inf")
    best_slope = mean_slope
    for s in np.arange(0.8 * mean_slope, 1.2 * mean_slope, 0.001 * mean_slope):
        fitted_line = s * x
        M = max(max(p - fitted_line) for p in phases)
        m = min(min(p - fitted_line) for p in phases)
        if M - m < remaining_variation:
            remaining_variation = M - m
            best_slope = s
    return best_slope * x, phases, best_slope


def fit_line_to_phase(fixed_inputs, B1, G, centers, half_width, count_slices_with_algorithm=False):
    n_slices = len(centers[0])
    n_b0_values = len(centers)
    mxy, mz = simulate_B0_values(fixed_inputs, B1, G, n_b0_values=n_b0_values)
    fitted_lines = []
    phases = []
    slopes = []
    for ff in range(mxy.shape[0]):
        phase = np.unwrap(np.angle(mxy[ff, :]))
        fitted_line = np.zeros_like(phase)

        (centers_comp, half_width_comp), success = count_slices(np.abs(mxy[ff, :]))
        if count_slices_with_algorithm:
            print("SLICES COUNTED!!!")
            slope, fitted_line, phase = fit_lines(phase, half_width_comp, centers_comp, n_slices, fitted_line)
        else:
            slope, fitted_line, phase = fit_lines(phase, half_width, centers[ff], n_slices, fitted_line)
        slopes.append(slope)
        fitted_lines.append(fitted_line)
        phases.append(phase)

    return fitted_lines, phases, slopes


def fit_lines(phase, half_width, centers_loc, n_slices, fitted_line):
    allPhases = np.zeros((2 * half_width + 1))
    x = np.arange(-half_width, half_width + 1, dtype=np.float32) / (2 * half_width) * 0.02
    for i in range(n_slices):
        center = centers_loc[i]
        start = max(center - half_width, 0)
        end = min(center + half_width, len(phase) - 1)
        allPhases += phase[start : end + 1] - np.mean(phase[start : end + 1])

    slope, intercept = np.polyfit(x, allPhases / n_slices, 1)
    for i in range(n_slices):
        center = centers_loc[i]
        start = max(center - half_width, 0)
        end = min(center + half_width, len(phase) - 1)
        fitted_line[start : end + 1] = slope * x + np.mean(phase[start : end + 1])
    return slope, fitted_line, phase


def plot_phase_fit_error2(fixed_inputs, B1, G, centers_allB0, half_width, path=None):
    n_b0_values = len(centers_allB0)
    fitted_line, phases, slope = fit_line_to_phase2(fixed_inputs, B1, G, centers_allB0, half_width)
    freq_offsets_ppm = torch.linspace(-water_ppm, 0.0, n_b0_values)
    cmap = cm.get_cmap("inferno", n_b0_values + 1)
    colors = [cmap(i) for i in range(n_b0_values)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, phase in enumerate(phases):
        label = f"{freq_offsets_ppm[i//2]:.1f} ppm" if i % 2 == 0 else None
        ax.plot(phase - fitted_line, linewidth=0.8, label=label, color=colors[i % n_b0_values])
    zero = phase[0] * 0.0
    ax.plot(zero, linewidth=0.8, color="black", linestyle="dotted")
    ax.set_title("Phase Fitting and Error")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Phase Value")
    ax.legend(fontsize="x-small")

    if path is not None:
        filename = "phase_error_on_targets.png"
        fig.savefig(os.path.join(os.path.dirname(path), filename), dpi=300)

    return [slope] * n_b0_values


def plot_phase_fit_error(fixed_inputs, B1, G, centers_allB0, half_width, count_slices_with_algorithm=False, path=None):
    n_b0_values = len(centers_allB0)
    fitted_lines, phases, slopes = fit_line_to_phase(fixed_inputs, B1, G, centers_allB0, half_width, count_slices_with_algorithm)
    freq_offsets_ppm = torch.linspace(-water_ppm, 0.0, n_b0_values)
    cmap = cm.get_cmap("inferno", n_b0_values + 1)
    colors = [cmap(i) for i in range(n_b0_values)]

    where_slices_are = np.zeros((n_b0_values, phases[0].shape[0]))
    for i, centers in enumerate(centers_allB0):
        for center in centers:
            where_slices_are[i, center - half_width : center + half_width] = 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fitted_line in enumerate(fitted_lines):
        error = phases[i] - fitted_line
        error[~where_slices_are[i].astype(bool)] = np.nan
        plt.plot(fixed_inputs["pos"], error, linewidth=0.8, label=f"{freq_offsets_ppm[i]:.1f} ppm", color=colors[i])
    zero = fixed_inputs["pos"] * 0.0
    zero[~where_slices_are.astype(bool).all(axis=0)] = np.nan
    plt.plot(fixed_inputs["pos"], zero, linewidth=0.8, color="black", linestyle="dotted")
    plt.title("Phase Fitting and Error")
    plt.xlabel("pos")
    plt.ylabel("Phase Value")
    plt.legend()

    if path is not None:
        filename = "phase_error_on_slices.png" if count_slices_with_algorithm else "phase_error_on_targets.png"
        fig.savefig(os.path.join(os.path.dirname(path), filename), dpi=300)

    return slopes
