import scipy
import torch
import numpy as np
from blochsim_CK import blochsim_CK


# BLOCH PARAMETERS:
def get_fixed_inputs(module=np, device=torch.device("cpu")):
    inputs = scipy.io.loadmat("data/blochInput.mat")
    inputs["returnallstates"] = False
    inputs["dt"] = inputs["dtmb"].item()
    dt = inputs["dt"]
    pos = torch.from_numpy(inputs["pos"]).to(torch.float32).detach().requires_grad_(False)
    inputs["pos"] = pos.to(device)
    Nz = 4096
    sens = module.ones((Nz, 1), dtype=module.complex64)
    B0 = module.zeros((Nz, 1)).to(device)
    tAx = module.linspace(0, (len(inputs["rfmb"]) - 1) * dt, Nz)
    bw = (1.0 / max(tAx)) * len(tAx)
    fAx = module.linspace(-bw, bw, len(tAx)) * 1e-6 * 2 * module.pi
    t_B1 = module.arange(0, len(inputs["rfmb"])) * dt * 1e3

    # Convert to float32
    if module.__name__ == "numpy":
        tAx = tAx.astype(np.float32)
        fAx = fAx.astype(np.float32)
        t_B1 = t_B1.astype(np.float32)
        return inputs, dt, Nz, sens, B0, G3, tAx, fAx, t_B1[:, np.newaxis]
    elif module.__name__ == "torch":
        tAx = tAx.float()
        fAx = fAx.float()
        t_B1 = t_B1.float()
        return inputs, dt, Nz, sens.to(device), B0, tAx, fAx, t_B1.unsqueeze(1).to(device)


def get_targets():
    inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch)
    G = torch.from_numpy(inputs["Gs"]).to(torch.float32)
    B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64)
    mxy, mz = blochsim_CK(B1=B1, G=G, sens=sens, B0=B0, **inputs)
    target_z = mz > 0.5
    target_xy = np.abs(mxy) > 0.5
    return target_z.to(torch.float32), target_xy.to(torch.float32)


# TRAINING PARAMETERS:
epochs = 1000
lr = 1e-7
gradient_scale = 100.0  # relative size of gradient to RF pulse
