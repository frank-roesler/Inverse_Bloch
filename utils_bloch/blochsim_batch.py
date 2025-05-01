import numpy as np
import torch
from utils_bloch.blochsim_CK import my_sinc


@torch.jit.script
def time_loop(
    Nt: int,
    reAlpha: torch.Tensor,
    reBeta: torch.Tensor,
    imAlpha: torch.Tensor,
    imBeta: torch.Tensor,
    reStatea: torch.Tensor,
    imStatea: torch.Tensor,
    reStateb: torch.Tensor,
    imStateb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Time loop for Bloch simulation, vectorized over batch dimension (Nb).

    Parameters
    ----------
    Nt : int
        Number of time points.
    reAlpha, reBeta, imAlpha, imBeta : torch.Tensor
        Real and imaginary parts of Cayley-Klein parameters (Nb x Ns x Nt).
    reStatea, imStatea, reStateb, imStateb : torch.Tensor
        Real and imaginary parts of state variables (Nb x Ns).

    Returns
    -------
    Updated real and imaginary parts of state variables (Nb x Ns).
    """
    for tt in range(Nt):
        retmpa = (
            reAlpha[:, :, tt] * reStatea
            - imAlpha[:, :, tt] * imStatea
            - (reBeta[:, :, tt] * reStateb + imBeta[:, :, tt] * imStateb)
        )
        imtmpa = (
            reAlpha[:, :, tt] * imStatea
            + imAlpha[:, :, tt] * reStatea
            - (reBeta[:, :, tt] * imStateb - imBeta[:, :, tt] * reStateb)
        )
        retmpb = (
            reBeta[:, :, tt] * reStatea
            - imBeta[:, :, tt] * imStatea
            + (reAlpha[:, :, tt] * reStateb + imAlpha[:, :, tt] * imStateb)
        )
        imtmpb = (
            reBeta[:, :, tt] * imStatea
            + imBeta[:, :, tt] * reStatea
            + (reAlpha[:, :, tt] * imStateb - imAlpha[:, :, tt] * reStateb)
        )
        reStatea = retmpa
        imStatea = imtmpa
        reStateb = retmpb
        imStateb = imtmpb
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

    # Initialize state variables
    statea = torch.ones((Nb, Ns), dtype=torch.complex64, device=B1.device, requires_grad=False)
    stateb = torch.zeros((Nb, Ns), dtype=torch.complex64, device=B1.device, requires_grad=False)

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
    Phi = torch.zeros(normSquared.shape, dtype=torch.float32, device=B1.device, requires_grad=False)
    Phi[normSquared > 0] = dt * gam * torch.sqrt(normSquared[normSquared > 0])
    sinc_part = -1j * gam * dt * 0.5 * my_sinc(Phi / 2)
    alpha = torch.cos(Phi / 2) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt

    # Loop over time
    reStatea, imStatea, reStateb, imStateb = time_loop(
        Nt,
        torch.real(alpha),
        torch.real(beta),
        torch.imag(alpha),
        torch.imag(beta),
        torch.real(statea),
        torch.imag(statea),
        torch.real(stateb),
        torch.imag(stateb),
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
    mz_batch = mz0 * (statea * stateaBar - stateb * statebBar) - 2 * torch.real(mxy0 * stateaBar * statebBar)

    return mxy_batch, mz_batch.real
