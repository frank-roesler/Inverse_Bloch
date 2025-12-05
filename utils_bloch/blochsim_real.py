import numpy as np
import torch
from utils_bloch.simulation_utils import sinc_of_root, cos_of_root, time_loop_realPulse
from constants import gamma as gam


def setup_simulation(G, pos, sens, B0_list, B1):
    Ns = pos.shape[0]  # Number of spatial positions
    Nb = B0_list.shape[0]  # Number of B0 values

    bxy = torch.matmul(sens, B1.T)  # Ns x Nt
    bz = pos * G  # Ns x Nt

    bz = bz.unsqueeze(0).repeat(Nb, 1, 1)  # Expand bz to shape (Nb, Ns, Nt)
    B0_list = B0_list.expand(-1, -1, bz.shape[2])  # Expand B0_list to shape (Nb, Ns, Nt)

    bz += B0_list  # Final shape: (Nb, Ns, Nt)
    return Nb, Ns, bxy, bz


def compute_alpha_beta_without_sqrt(bxy, bz, dt):
    Phi_half_squared = (0.5 * dt * gam) ** 2 * (bxy**2 + bz**2)  # Nb x Ns x Nt
    sinc_part = -gam * dt * 0.5 * sinc_of_root(Phi_half_squared)
    reAlpha = cos_of_root(Phi_half_squared)
    imAlpha = -bz * sinc_part
    imBeta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt
    return reAlpha, imAlpha, imBeta


def blochsim_CK_batch_realPulse(B1, G, pos, sens, B0_list, M0, dt=6.4e-6):
    Nb, Ns, bxy, bz = setup_simulation(G, pos.real, sens.real, B0_list, B1)

    reAlpha, imAlpha, imBeta = compute_alpha_beta_without_sqrt(bxy, bz, dt)

    # Loop over time
    reStatea, imStatea, reStateb, imStateb = time_loop_realPulse(
        reAlpha,
        imAlpha,
        imBeta,
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
