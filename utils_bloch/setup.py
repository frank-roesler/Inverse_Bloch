import scipy
import torch
import numpy as np
from utils_bloch.blochsim_CK import blochsim_CK
from buildTarget import buildTarget


# BLOCH PARAMETERS:
def get_fixed_inputs():
    Nz = 4096
    inputs = scipy.io.loadmat("data/smPulse_512pts.mat")
    inputs["returnallstates"] = False
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
    dx = (inputs["pos"][-1, 2] - inputs["pos"][0, 2]) / (len(inputs["pos"][:, 2]) - 1)
    inputs["pos"] = torch.from_numpy(inputs["pos"]).to(torch.float32).detach().requires_grad_(False)
    sens = sens.detach().requires_grad_(False)
    B0 = B0.detach().requires_grad_(False)
    tAx = tAx.detach().requires_grad_(False)
    fAx = fAx.detach().requires_grad_(False)
    return (inputs, dt, dx.item(), Nz, sens, B0, tAx, fAx, t_B1.unsqueeze(1))


def get_test_targets():
    inputs, dt, dx, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
    pos = inputs["pos"]
    target_xy = torch.zeros((len(fAx)), dtype=torch.float32, requires_grad=False)
    target_xy[pos[:, 2] > -0.025] = 1
    target_xy[pos[:, 2] > -0.005] = 0
    target_xy[pos[:, 2] > 0.005] = 1
    target_xy[pos[:, 2] > 0.025] = 0
    target_z = 1 - target_xy
    return target_z, target_xy


def get_smooth_targets(smoothness=1):
    inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
    G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
    B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
    mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)
    targAbsMxy, targPhMxy, targMz = buildTarget(mxy, mz, B1, dt, bwTB=1.8, sharpnessTB=smoothness, phSlopReduction=1)
    targAbsMxy = torch.from_numpy(targAbsMxy).to(torch.float32).requires_grad_(False)
    targPhMxy = torch.from_numpy(targPhMxy).to(torch.float32).requires_grad_(False)
    targMz = torch.from_numpy(targMz).to(torch.float32).requires_grad_(False)
    return (targAbsMxy, targPhMxy, targMz)
