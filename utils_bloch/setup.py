import scipy
import torch
import numpy as np
from utils_bloch.blochsim_CK import blochsim_CK
from buildTarget import buildTarget


# BLOCH PARAMETERS:
def get_fixed_inputs(tfactor=1.0):
    Nz = 4096
    inputs = scipy.io.loadmat("data/smPulse_512pts.mat")
    inputs["returnallstates"] = False
    inputs["dtmb"] *= tfactor
    inputs["dt"] = inputs["dtmb"].item()
    dt = inputs["dt"]
    sens = torch.ones((Nz, 1), dtype=torch.complex64)
    B0 = torch.zeros((Nz, 1))
    tAx = torch.linspace(0, (len(inputs["rfmb"]) - 1) * dt, Nz)
    bw = len(tAx) / torch.max(tAx)
    fAx = torch.linspace(-bw, bw, len(tAx)) * 1e-6 * 2 * torch.pi
    t_B1 = torch.arange(0, len(inputs["rfmb"])) * dt * 1e3
    tAx = tAx.float()
    fAx = fAx.float()
    t_B1 = t_B1.float()
    pos = torch.from_numpy(inputs["pos"]).to(torch.float32).detach().requires_grad_(False)
    dx = (pos[-1, 2] - pos[0, 2]) / (len(pos[:, 2]) - 1)
    inputs["pos"] = pos
    sens = sens.detach().requires_grad_(False)
    B0 = B0.detach().requires_grad_(False)
    tAx = tAx.detach().requires_grad_(False)
    fAx = fAx.detach().requires_grad_(False)
    M0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    pos = pos.contiguous()
    sens = sens.contiguous()
    B0 = B0.contiguous()
    t_B1 = t_B1.contiguous()
    M0 = M0.contiguous()
    return (pos, dt, dx.item(), Nz, sens, B0, tAx, fAx, t_B1.unsqueeze(1), M0, inputs)


def get_targets(theta=0.0):
    import params

    target_xy = torch.zeros((len(params.fAx)), dtype=torch.float32, requires_grad=False)
    target_xy[params.pos[:, 2] > -0.025] = np.sin(theta)
    target_xy[params.pos[:, 2] > -0.005] = 0
    target_xy[params.pos[:, 2] > 0.005] = np.sin(theta)
    target_xy[params.pos[:, 2] > 0.025] = 0
    target_z = torch.sqrt(1 - target_xy**2)
    return target_z, target_xy


# def get_smooth_targets(smoothness=1):
#     pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs = get_fixed_inputs()
#     # G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
#     # B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
#     mz, mxy = get_targets(np.pi / 2)
#     # mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)
#     targAbsMxy, targPhMxy, targMz = buildTarget(mxy, mz, fAx, dt, bwTB=0.1, sharpnessTB=smoothness, phSlopReduction=1)
#     targAbsMxy = torch.from_numpy(targAbsMxy).to(torch.float32).requires_grad_(False)
#     targPhMxy = torch.from_numpy(targPhMxy).to(torch.float32).requires_grad_(False)
#     targMz = torch.from_numpy(targMz).to(torch.float32).requires_grad_(False)
#     return targAbsMxy, targPhMxy, targMz


import torch


def smooth_square_well_atan(x, left=-0.5, right=0.5, depth=1.0, smoothness=10.0):
    """
    Smooth approximation of a square well function using torch.atan with specified boundaries.

    Parameters:
        x (torch.Tensor): Input tensor.
        left (float): Left boundary of the square well.
        right (float): Right boundary of the square well.
        depth (float): Depth of the square well (negative value for a well).
        smoothness (float): Controls the smoothness of the edges (higher = sharper).

    Returns:
        torch.Tensor: Smooth square well function values.
    """
    left_transition = 0.5 * (1 + 2 * torch.atan(smoothness * (x - left)) / torch.pi)
    right_transition = 0.5 * (1 + 2 * torch.atan(smoothness * (right - x)) / torch.pi)
    well = depth * (left_transition * right_transition)
    return well


def get_smooth_targets(theta=np.pi / 2, smoothness=1):
    import params

    left_slice = smooth_square_well_atan(
        params.pos[:, 2], left=-0.025, right=-0.005, depth=np.sin(theta), smoothness=smoothness
    )
    right_slice = smooth_square_well_atan(
        params.pos[:, 2], right=0.025, left=0.005, depth=np.sin(theta), smoothness=smoothness
    )
    target_xy = left_slice + right_slice
    target_z = torch.sqrt(1 - target_xy**2)
    return target_z, target_xy
