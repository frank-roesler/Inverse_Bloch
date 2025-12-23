import torch
import numpy as np


def smooth_square_well(x, left=-0.5, right=0.5, depth=1.0, smoothness=10.0, function=torch.sigmoid):
    step_left = function(smoothness * (x - left))
    step_right = function(smoothness * (right - x))
    left_transition = step_left
    right_transition = step_right
    if function == torch.atan:
        left_transition = 0.5 * (1 + 2 * step_left / torch.pi)
        right_transition = 0.5 * (1 + 2 * step_right / torch.pi)
    well = depth * (left_transition * right_transition)
    return well


def circshift(x, shift):
    shift = shift % x.numel()
    if shift == 0:
        return x
    return torch.cat((x[-shift:], x[:-shift]))


def get_smooth_targets(train_config, bloch_config, function=torch.sigmoid, override_inputs=None):
    """Creates target |Mxy| and Mz profiles. Mz profile is real, while |Mxy| profile is artificially made complex to encode
    the slice number. Each slice is multiplied by a different complex root of unity. The actual profile is obtained by taking
    abs(target_xy).
    - train_config.target_smoothness: Higher values give sharper transitions
    - train_config.shift_targets: If true, shifts target profiles for different B0 offsets to reflect chemical shift
    """
    smoothness = train_config.target_smoothness * 1000.0
    width = 0.02
    distance = 0.01
    shift = 0.001 if train_config.shift_targets else 0.0

    pos = bloch_config.fixed_inputs["pos"] if override_inputs is None else override_inputs["pos"]

    target_xy = torch.zeros((bloch_config.n_b0_values, pos.shape[0], bloch_config.n_slices), dtype=torch.float32, requires_grad=False)
    minShift = bloch_config.n_b0_values // 2
    for b in range(-minShift, minShift + 1):
        left = -0.5 * (width * bloch_config.n_slices + distance * (bloch_config.n_slices - 1)) - b * shift
        centers_loc = []
        for s in range(bloch_config.n_slices):
            target_xy[b, :, s] = smooth_square_well(
                pos,
                left=left,
                right=left + width,
                depth=np.sin(bloch_config.flip_angle),
                smoothness=smoothness,
                function=function,
            )
            centers_loc.append(np.argmin(np.abs(pos - (left + 0.5 * width))).item())
            left += width + distance

    return target_xy
