import numpy as np
import torch
from utils_bloch.simulation_utils import *


def blochsim_CK_batch(B1, G, pos, sens, B0_list, M0, dt=6.4e-6):
    """
    Batch version of Bloch Simulator using Cayley-Klein parameters for multiple voxels simultaneously.

    Parameters
    ----------
    B1 : torch.Tensor
        B1(t,c) = Nt x Nc tensor (RF pulse waveform for each coil)
    G : torch.Tensor
        [Gx(t) Gy(t) Gz(t)] = Nt x 3 tensor (Gradient waveforms)
    pos : torch.Tensor
        [x(:) y(:) z(:)] = Ns x 3 tensor (Spatial positions)
    sens : torch.Tensor
        [S1(:) S2(:) ...Sn(:)] = Ns x Nc tensor (Coil sensitivities)
    B0_list : torch.Tensor
        B0_list(k,:) = Ns x 1 tensor for each B0 value (Off-resonance field for each simulation)
    M0 : torch.Tensor
        Initial magnetization (default [0,0,1])
    dt : float
        Time step in seconds (default 6.4e-6)

    Returns
    -------
    mxy_batch : torch.Tensor
        Final transverse magnetization for each B0 value
    mz_batch : torch.Tensor
        Final longitudinal magnetization for each B0 value
    """
    gam, Nb, Ns, Nt, bxy, bz = setup_simulation(G, pos, sens, B0_list, B1)

    alpha, beta = compute_alpha_beta(bxy, bz, dt, gam)

    # Loop over time
    reStatea, imStatea, reStateb, imStateb = time_loop_real(
        torch.real(alpha),
        torch.real(beta),
        torch.imag(alpha),
        torch.imag(beta),
        Nb,
        Ns,
        B1.device,
    )
    statea, stateb = reStatea + 1j * imStatea, reStateb + 1j * imStateb
    stateaBar, statebBar = reStatea - 1j * imStatea, reStateb - 1j * imStateb

    # Calculate final magnetization state (M0 can be 3x1 or 3xNs)
    if M0.ndim == 1:
        mxy0 = torch.complex(M0[0], M0[1])
        mz0 = M0[2]

        # Broadcast to match Ns
        mxy0 = mxy0.expand(Nb, Ns)
        mz0 = mz0.expand(Nb, Ns)
    else:
        if M0.shape[0] == 3:
            mxy0 = torch.complex(M0[0], M0[1])
            mz0 = M0[2]
        else:
            mxy0 = torch.complex(M0[:, 0], M0[:, 1])
            mz0 = M0[:, 2]

    # Calculate final magnetization
    mxy_batch = 2 * mz0 * stateaBar * stateb + mxy0 * stateaBar**2 - torch.conj(mxy0) * stateb**2
    mz_batch = mz0 * (statea * stateaBar).real - (stateb * statebBar).real - 2 * (mxy0 * stateaBar * statebBar).real

    return mxy_batch, mz_batch.real
