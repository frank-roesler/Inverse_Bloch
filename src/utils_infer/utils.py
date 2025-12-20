import numpy as np


def compute_phase_offsets(mxy_ff, slice_centers_current_b0, half_width):
    phase = np.angle(mxy_ff)
    phases_current_b0 = []
    phase_offsets_current_b0 = []
    for c in slice_centers_current_b0:
        phase_loc = phase[c - half_width : c + half_width]
        phases_current_b0.append(phase_loc)

    if len(phases_current_b0) > 1:
        for j in range(len(phases_current_b0) - 1):
            phasediffexp = np.exp(1j * phases_current_b0[j + 1]) / np.exp(1j * phases_current_b0[j])
            phase_offsets_current_b0.append(np.mean(np.angle(phasediffexp)))
    return phase_offsets_current_b0
