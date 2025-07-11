import scipy
import torch
import numpy as np

# from utils_bloch.blochsim_CK import blochsim_CK
# from buildTarget import buildTarget


# BLOCH PARAMETERS:
def get_fixed_inputs(tfactor=1.0, n_b0_values=1, Nz=4096, Nt=512, pos_spacing="linear", n_slices=1):
    gam = 267522.1199722082
    gam_hz_mt = gam / (2 * np.pi)
    inputs = scipy.io.loadmat("data/smPulse_512pts.mat")
    inputs["returnallstates"] = False
    inputs["dt"] = inputs["dtmb"].item()
    dt = inputs["dt"]
    sens = torch.ones((Nz, 1), dtype=torch.complex64)
    B0 = torch.zeros((Nz, 1))
    Tmin = dt * 1e3 * 512
    T = Tmin * tfactor
    Nt = int(Nt * tfactor)
    t_B1 = torch.linspace(0, T, Nt)
    t_B1 = t_B1.float()
    dt_num = (t_B1[-1] - t_B1[0]) / (len(t_B1) - 1)

    if pos_spacing == "nonlinear":
        u = torch.linspace(-1, 1, Nz)
        pos = abs(u) ** 3
        pos[u < 0] = -pos[u < 0]
        pos += 0.25 * n_slices * u
        pos = 2 * 0.18 * pos / (torch.max(pos) - torch.min(pos))
        pos = pos.detach()
    else:
        pos = torch.linspace(-0.18, 0.18, Nz)

    dx = (pos[-1] - pos[0]) / (len(pos) - 1)
    inputs["pos"] = pos
    sens = sens.detach().requires_grad_(False)
    B0 = B0.detach().requires_grad_(False)
    M0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    if n_b0_values == 1:
        freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, 2)
        freq_offsets_Hz = torch.mean(freq_offsets_Hz, dim=0, keepdim=True)
    else:
        freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, n_b0_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
    B0_vals = []
    for ff in range(len(freq_offsets_Hz)):
        B0_vals.append(B0 + B0_freq_offsets_mT[ff])
    B0_list = torch.stack(B0_vals, dim=0).to(torch.float32)
    return {
        "pos": pos,  # [dm]
        "dt": dt * tfactor,
        "dt_num": dt_num.item() * 1e-3,
        "dx": dx.item(),
        "Nz": Nz,
        "sens": sens,
        "B0": B0,
        "t_B1": t_B1.unsqueeze(1),
        "t_B1_legacy": torch.arange(0, len(inputs["rfmb"])).unsqueeze(1) * dt * 1e3 * tfactor,
        "M0": M0,
        "inputs": inputs,
        "freq_offsets_Hz": freq_offsets_Hz,
        "B0_list": B0_list,
        "gam": gam,
        "gam_hz_mt": gam_hz_mt,
    }


def get_targets(theta=0.0):
    import params

    target_xy = torch.zeros((len(params.pos)), dtype=torch.float32, requires_grad=False)
    target_xy[params.pos > -0.025] = np.sin(theta)
    target_xy[params.pos > -0.005] = 0
    target_xy[params.pos > 0.005] = np.sin(theta)
    target_xy[params.pos > 0.025] = 0
    target_z = torch.sqrt(1 - target_xy**2)
    return target_z, target_xy


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


def get_smooth_targets(theta=np.pi / 2, smoothness=1, function=torch.sigmoid, n_targets=1, pos=torch.linspace(-0.18, 0.18, 4096), n_b0_values=1, shift_targets=False):
    """higher smoothness values give sharper transitions"""

    smoothness *= 1000.0

    width = 0.02
    distance = 0.01
    shift = 0.002 if shift_targets else 0.0
    pos0 = np.argmin(np.abs(pos)).item()
    posAtWidth = np.argmin(np.abs(pos - width / 2)).item()
    half_width = posAtWidth - pos0

    targets_xy = []
    centers = []
    minShift = n_b0_values // 2
    for i in range(-minShift, minShift + 1):

        left = -0.5 * (width * n_targets + distance * (n_targets - 1)) - i * shift

        target_xy = torch.zeros(pos.shape, dtype=torch.float32, requires_grad=False)
        centers_loc = []
        for i in range(n_targets):
            target_xy += smooth_square_well(
                pos,
                left=left,
                right=left + width,
                depth=np.sin(theta),
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


# fixed_inputs = get_fixed_inputs(tfactor=2, n_b0_values=3, Nz=128, Nt=64, pos_spacing="nonlinear", n_slices=4)
# pos = fixed_inputs["pos"]
# import matplotlib.pyplot as plt

# plt.plot(pos, 0 * pos, ".", markersize=2)
# plt.show()
