import numpy as np
import torch
import os


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


def export_as_numpy(torch_path, output_dir):
    data = torch.load(torch_path, weights_only=False)
    pulse = data["pulse"].cpu().numpy()
    gradient = data["gradient"].cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "pulse_converted_numpy.npy"), pulse)
    np.save(os.path.join(output_dir, "gradient_converted_numpy.npy"), gradient)
    print(f"Saved pulse and gradient to {output_dir}")
