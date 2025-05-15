import numpy as np
import torch
from utils_bloch.blochsim_CK import my_sinc


def time_loop(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    Nb: int,
    Ns: int,
    Nt: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    statea = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    stateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)

    for tt in range(Nt - 1):
        alpha_t = alpha[:, :, tt]
        beta_t = beta[:, :, tt]
        statea_t = statea[:, :, tt]
        stateb_t = stateb[:, :, tt]
        statea[:, :, tt + 1] = alpha_t * statea_t - torch.conj(beta_t) * stateb_t
        stateb[:, :, tt + 1] = beta_t * statea_t + torch.conj(alpha_t) * stateb_t
    return (statea, stateb)


def blochsim_CK_timeprofile(B1, G, pos, sens, B0_list, M0, dt=6.4e-6):

    # Constants
    G = torch.column_stack((0 * G.flatten(), 0 * G.flatten(), G.flatten()))
    gam = 267522.1199722082  # radians per sec per mT

    # Get dimensions
    Ns = pos.shape[0]  # Number of spatial positions
    Nt = G.shape[0]  # Number of time points
    Nb = B0_list.shape[0]  # Number of B0 values

    # Sum up RF over coils: bxy = sens * B1.T
    bxy = torch.matmul(sens, B1.T)  # Ns x Nt
    # Sum up gradient over channels: bz = pos * G.T
    bz = torch.matmul(pos, G.T)  # Ns x Nt

    # Add off-resonance for each B0 value
    bz = bz.unsqueeze(0).repeat(Nb, 1, 1)  # Expand bz to shape (Nb, Ns, Nt)
    B0_list = B0_list.expand(-1, -1, bz.shape[2])  # Expand B0_list to shape (Nb, Ns, Nt)

    # Add in chunks to avoid excessive memory usage
    bz += B0_list  # Final shape: (Nb, Ns, Nt)

    # Compute these out of loop
    normSquared = (bxy * torch.conj(bxy)).real + bz**2  # Nb x Ns x Nt
    posNormPts = normSquared > 0
    Phi = torch.zeros(normSquared.shape, dtype=torch.float32, device=B1.device, requires_grad=False)
    Phi[posNormPts] = dt * gam * torch.sqrt(normSquared[posNormPts])
    sinc_part = -1j * gam * dt * 0.5 * my_sinc(Phi / 2)
    alpha = torch.cos(Phi / 2) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt

    # Loop over time
    statea, stateb = time_loop(
        alpha,
        beta,
        Nb,
        Ns,
        Nt,
        B1.device,
    )
    stateaBar, statebBar = torch.conj(statea), torch.conj(stateb)

    # Calculate final magnetization state (M0 can be 3x1 or 3xNs)
    if M0.ndim == 1:
        mxy0 = torch.complex(M0[0], M0[1])
        mz0 = M0[2]

        # Broadcast to match Ns
        mxy0 = mxy0.expand(Nb, Ns, Nt)
        mz0 = mz0.expand(Nb, Ns, Nt)
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
