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


def get_smooth_targets(bloch_config, train_config, function=torch.sigmoid):
    """higher smoothness values give sharper transitions"""

    smoothness = train_config.target_smoothness * 1000.0
    width = 0.02
    distance = 0.01
    shift = 0.002 if train_config.shift_targets else 0.0

    pos = bloch_config.fixed_inputs["pos"]
    pos0 = np.argmin(np.abs(pos)).item()
    posAtWidth = np.argmin(np.abs(pos - width / 2)).item()
    half_width = posAtWidth - pos0

    targets_xy = []
    centers = []
    minShift = bloch_config.n_b0_values // 2
    for i in range(-minShift, minShift + 1):

        left = -0.5 * (width * bloch_config.n_slices + distance * (bloch_config.n_slices - 1)) - i * shift

        target_xy = torch.zeros(pos.shape, dtype=torch.float32, requires_grad=False)
        centers_loc = []
        for i in range(bloch_config.n_slices):
            target_xy += smooth_square_well(
                pos,
                left=left,
                right=left + width,
                depth=np.sin(bloch_config.flip_angle),
                smoothness=smoothness,
                function=function,
            )
            centers_loc.append(np.argmin(np.abs(pos - (left + 0.5 * width))).item())
            left += width + distance

        targets_xy.append(target_xy)
        centers.append(centers_loc)
    target_xy = torch.stack(targets_xy, dim=0)
    target_z = torch.sqrt(1 - target_xy**2)
    return target_z, target_xy, centers, half_width
