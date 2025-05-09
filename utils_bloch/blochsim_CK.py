"""
Bloch Simulator using Cayley-Klein parameters for multiple voxels simultaneously.
Python implementation with NumPy that's compatible with PyTorch tensors.

Adapted from MATLAB code by SJM and Will Grissom (http://www.vuiis.vanderbilt.edu/~grissowa/)

This implementation works with both NumPy arrays and PyTorch tensors through a
compatibility layer that detects the input type and uses the appropriate functions.
"""

import numpy as np
import torch


def my_sinc(x):
    if x.device != torch.device("mps:0"):
        return torch.sinc(x / np.pi)
    large_x = torch.abs(x) > 1e-2
    result = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    result[~large_x] = 1 - x[~large_x] ** 2 / 6 + x[~large_x] ** 4 / 120 + x[~large_x] ** 6 / 5040
    result[large_x] = torch.sin(x[large_x]) / x[large_x]
    return result


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
    for tt in range(Nt):
        retmpa = (
            reAlpha[:, tt] * reStatea
            - imAlpha[:, tt] * imStatea
            - (reBeta[:, tt] * reStateb + imBeta[:, tt] * imStateb)
        )
        imtmpa = (
            reAlpha[:, tt] * imStatea
            + imAlpha[:, tt] * reStatea
            - (reBeta[:, tt] * imStateb - imBeta[:, tt] * reStateb)
        )
        retmpb = (
            reBeta[:, tt] * reStatea
            - imBeta[:, tt] * imStatea
            + (reAlpha[:, tt] * reStateb + imAlpha[:, tt] * imStateb)
        )
        imtmpb = (
            reBeta[:, tt] * imStatea
            + imBeta[:, tt] * reStatea
            + (reAlpha[:, tt] * imStateb - imAlpha[:, tt] * reStateb)
        )
        # tmpa = alpha[:, tt] * statea - betaBar[:, tt] * stateb
        # stateb = beta[:, tt] * statea + alphaBar[:, tt] * stateb
        reStatea = retmpa
        imStatea = imtmpa
        reStateb = retmpb
        imStateb = imtmpb
    return (reStatea, imStatea, reStateb, imStateb)


def blochsim_CK(B1, G, pos, sens, B0, M0, dt=6.4e-6):
    """
    Bloch Simulator using Cayley-Klein parameters for multiple voxels simultaneously.

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
    B0 : torch.Tensor
        B0(:) = Ns x 1 tensor (Off-resonance field)
    **kwargs : dict
        Optional parameters:
        - dt: time step in seconds (default 6.4e-6)
        - M0: initial magnetization (default [0,0,1])

    Returns
    -------
    mxy : torch.Tensor
        Final transverse magnetization
    mz : torch.Tensor
        Final longitudinal magnetization
    mxyt : torch.Tensor, optional
        Transverse magnetization over time (if returnallstate is True)
    mzt : torch.Tensor, optional
        Longitudinal magnetization over time (if returnallstate is True)
    a : torch.Tensor
        Cayley-Klein parameter alpha (final value or time series)
    b : torch.Tensor
        Cayley-Klein parameter beta (final value or time series)
    """
    # Constants
    G = torch.column_stack((0 * G.flatten(), 0 * G.flatten(), G.flatten()))
    gam = 267522.1199722082  # radians per sec per mT

    # Get dimensions
    Ns = pos.shape[0]  # Number of spatial positions
    Nt = G.shape[0]  # Number of time points

    # Initialize state variables
    statea = torch.ones(Ns, dtype=torch.complex64, device=B1.device, requires_grad=False)
    stateb = torch.zeros(Ns, dtype=torch.complex64, device=B1.device, requires_grad=False)

    # Sum up RF over coils: bxy = sens * B1.T
    bxy = torch.matmul(sens, B1.T)

    # Sum up gradient over channels: bz = pos * G.T
    bz = torch.matmul(pos, G.T)

    # Add off-resonance
    if B0.ndim == 1:
        B0 = B0.unsqueeze(1)
    bz = bz + B0.repeat(1, Nt)

    # Compute these out of loop
    normSquared = torch.abs(bxy) ** 2 + bz**2
    Phi = torch.zeros(normSquared.shape, dtype=torch.float32, device=B1.device, requires_grad=False)
    Phi[normSquared > 0] = dt * gam * torch.sqrt(normSquared[normSquared > 0])
    sinc_part = -1j * gam * dt * 0.5 * my_sinc(Phi / 2)
    alpha = torch.cos(Phi / 2) - bz * sinc_part
    beta = -bxy * sinc_part

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
        mxy0 = mxy0.expand(Ns)
        mz0 = mz0.expand(Ns)
    else:
        if M0.shape[0] == 3:
            mxy0 = torch.complex(M0[0], M0[1])
            mz0 = M0[2]
        else:
            mxy0 = torch.complex(M0[:, 0], M0[:, 1])
            mz0 = M0[:, 2]

    # Calculate final magnetization
    mxy = 2 * mz0 * stateaBar * stateb + mxy0 * stateaBar**2 - torch.conj(mxy0) * stateb**2
    mz = mz0 * (statea * stateaBar - stateb * statebBar) - 2 * torch.real(mxy0 * stateaBar * statebBar)

    return mxy, mz.real
