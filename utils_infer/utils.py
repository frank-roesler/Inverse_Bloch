import numpy as np


def compute_phase_offsets(phase, slice_centers_current_b0, half_width):
    phase_means_current_b0 = []
    phase_offsets_current_b0 = []
    for c in slice_centers_current_b0:
        phase_loc = phase[c - half_width : c + half_width]
        phase_mean_slice = np.mean(phase_loc)
        phase_means_current_b0.append(phase_mean_slice)
        if len(phase_means_current_b0) > 1:
            phase_offsets_current_b0 = [phase_means_current_b0[j + 1] - phase_means_current_b0[j] for j in range(len(phase_means_current_b0) - 1)]
    return phase_offsets_current_b0
