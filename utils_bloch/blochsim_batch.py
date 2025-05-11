import numpy as np
import torch
from utils_bloch.blochsim_CK import my_sinc


@torch.jit.script
def time_loop(
    reAlpha: torch.Tensor,
    reBeta: torch.Tensor,
    imAlpha: torch.Tensor,
    imBeta: torch.Tensor,
    Nb: int,
    Ns: int,
    Nt: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reStatea = torch.ones((Nb, Ns, Nt), dtype=torch.float32, device=device)
    reStateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    imStatea = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    imStateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    for tt in range(Nt - 1):
        reStatea[:, :, tt + 1] = reAlpha[:, :, tt] * reStatea[:, :, tt] - imAlpha[:, :, tt] * imStatea[:, :, tt] - (reBeta[:, :, tt] * reStateb[:, :, tt] + imBeta[:, :, tt] * imStateb[:, :, tt])
        imStatea[:, :, tt + 1] = reAlpha[:, :, tt] * imStatea[:, :, tt] + imAlpha[:, :, tt] * reStatea[:, :, tt] - (reBeta[:, :, tt] * imStateb[:, :, tt] - imBeta[:, :, tt] * reStateb[:, :, tt])
        reStateb[:, :, tt + 1] = reBeta[:, :, tt] * reStatea[:, :, tt] - imBeta[:, :, tt] * imStatea[:, :, tt] + (reAlpha[:, :, tt] * reStateb[:, :, tt] + imAlpha[:, :, tt] * imStateb[:, :, tt])
        imStateb[:, :, tt + 1] = reBeta[:, :, tt] * imStatea[:, :, tt] + imBeta[:, :, tt] * reStatea[:, :, tt] + (reAlpha[:, :, tt] * imStateb[:, :, tt] - imAlpha[:, :, tt] * reStateb[:, :, tt])
    return (reStatea, imStatea, reStateb, imStateb)


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
    normSquared = torch.abs(bxy) ** 2 + bz**2  # Nb x Ns x Nt
    posNormPts = normSquared > 0
    Phi = torch.zeros(normSquared.shape, dtype=torch.float32, device=B1.device, requires_grad=False)
    Phi[posNormPts] = dt * gam * torch.sqrt(normSquared[posNormPts])
    sinc_part = -1j * gam * dt * 0.5 * my_sinc(Phi / 2)
    alpha = torch.cos(Phi / 2) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt

    # Loop over time
    reStatea, imStatea, reStateb, imStateb = time_loop(
        torch.real(alpha),
        torch.real(beta),
        torch.imag(alpha),
        torch.imag(beta),
        Nt,
        Ns,
        Nb,
        B1.device,
    )
    statea, stateb = reStatea + 1j * imStatea, reStateb + 1j * imStateb
    stateaBar, statebBar = reStatea - 1j * imStatea, reStateb - 1j * imStateb

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
    mz_batch = mz0 * (statea * stateaBar - stateb * statebBar) - 2 * torch.real(mxy0 * stateaBar * statebBar)

    return mxy_batch, mz_batch.real
