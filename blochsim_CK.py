"""
Bloch Simulator using Cayley-Klein parameters for multiple voxels simultaneously.
Python implementation with NumPy that's compatible with PyTorch tensors.

Adapted from MATLAB code by SJM and Will Grissom (http://www.vuiis.vanderbilt.edu/~grissowa/)

This implementation works with both NumPy arrays and PyTorch tensors through a
compatibility layer that detects the input type and uses the appropriate functions.
"""

import numpy as np
import torch
from typing import Tuple, Union, List, Optional, Any, Dict


G3 = lambda Gz: torch.column_stack((0 * Gz.flatten(), 0 * Gz.flatten(), Gz.flatten()))


def blochsim_CK(B1, G, pos, sens, B0, **kwargs):
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
    G = G3(G)
    gam = 267522.1199722082  # radians per sec per mT
    dt = kwargs.get("dt", 6.4e-6)

    # Handle M0 initialization
    M0 = kwargs.get("M0", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, requires_grad=False))
    if M0.ndim == 1:
        M0 = M0.to(B1.device)

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
    Phi = dt * gam * torch.sqrt(torch.abs(bxy) ** 2 + bz**2)
    cp = torch.cos(Phi / 2)
    sinc_part = 1j * gam * dt * 0.5 * (1 - Phi**2 / 24)
    alpha = cp - bz * sinc_part
    beta = -bxy * sinc_part
    alphaBar = torch.conj(alpha)
    betaBar = torch.conj(beta)

    # Loop over time
    for tt in range(Nt):
        tmpa = alpha[:, tt] * statea - betaBar[:, tt] * stateb
        stateb = beta[:, tt] * statea + alphaBar[:, tt] * stateb
        statea = tmpa

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
    mxy = 2 * mz0 * torch.conj(statea) * stateb + mxy0 * torch.conj(statea) ** 2 - torch.conj(mxy0) * stateb**2
    mz = mz0 * (statea * torch.conj(statea) - stateb * torch.conj(stateb))
    mz = mz + 2 * torch.real(mxy0 * torch.conj(statea) * (-torch.conj(stateb)))

    return mxy, mz.real
