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


def _is_torch_tensor(x):
    """Check if input is a PyTorch tensor without importing torch."""
    return x.__class__.__module__.startswith("torch")


def _get_module(x):
    """Get the appropriate module (numpy or torch) based on input type."""
    if _is_torch_tensor(x):
        import torch

        return torch
    return np


def _conj(x, module=None):
    """Get complex conjugate using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.conj(x)


def _real(x, module=None):
    """Get real part using the appropriate module."""
    if module is None:
        module = _get_module(x)
    if module.__name__ == "torch":
        return module.real(x)
    return module.real(x)


def _complex(real, imag, module=np):
    """Create complex number using the appropriate module."""
    if module.__name__ == "torch":
        return module.complex(real, imag)
    return real + 1j * imag


def _zeros_like(x, module=None):
    """Create zeros array like x using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.zeros_like(x)


def _ones_like(x, module=None):
    """Create ones array like x using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.ones_like(x)


def _zeros(shape, dtype=None, module=np):
    """Create zeros array using the appropriate module."""
    return module.zeros(shape, dtype=dtype)


def _ones(shape, dtype=None, module=np):
    """Create ones array using the appropriate module."""
    return module.ones(shape, dtype=dtype)


def _sqrt(x, module=None):
    """Calculate square root using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.sqrt(x)


def _cos(x, module=None):
    """Calculate cosine using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.cos(x)


def _sin(x, module=None):
    """Calculate sine using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.sin(x)


def _abs(x, module=None):
    """Calculate absolute value using the appropriate module."""
    if module is None:
        module = _get_module(x)
    return module.abs(x)


def _isfinite(x, module=None):
    """Check if values are finite using the appropriate module."""
    if module is None:
        module = _get_module(x)
    if module.__name__ == "torch":
        return module.isfinite(x)
    return module.isfinite(x)


def _where(condition, x, y, module=None):
    """Conditional selection using the appropriate module."""
    if module is None:
        if _is_torch_tensor(condition) or _is_torch_tensor(x) or _is_torch_tensor(y):
            import torch

            module = torch
        else:
            module = np
    return module.where(condition, x, y)


def _transpose(x, module=None):
    """Transpose the input using the appropriate module."""
    if module is None:
        module = _get_module(x)
    if module.__name__ == "torch":
        return module.transpose(x, 0, 1)
    return module.transpose(x)


# def blochsim_CK(B1, G, pos, sens, B0, **kwargs):
#     """
#     Bloch Simulator using Cayley-Klein parameters for multiple voxels simultaneously.

#     Parameters
#     ----------
#     B1 : array_like
#         B1(t,c) = Nt x Nc array (RF pulse waveform for each coil)
#     G : array_like
#         [Gx(t) Gy(t) Gz(t)] = Nt x 3 array (Gradient waveforms)
#     pos : array_like
#         [x(:) y(:) z(:)] = Ns x 3 array (Spatial positions)
#     sens : array_like
#         [S1(:) S2(:) ...Sn(:)] = Ns x Nc array (Coil sensitivities)
#     B0 : array_like
#         B0(:) = Ns x 1 vector (Off-resonance field)
#     **kwargs : dict
#         Optional parameters:
#         - dt: time step in seconds (default 6.4e-6)
#         - M0: initial magnetization (default [0,0,1])

#     Returns
#     -------
#     mxy : array_like
#         Final transverse magnetization
#     mz : array_like
#         Final longitudinal magnetization
#     mxyt : array_like, optional
#         Transverse magnetization over time (if returnallstate is True)
#     mzt : array_like, optional
#         Longitudinal magnetization over time (if returnallstate is True)
#     a : array_like
#         Cayley-Klein parameter alpha (final value or time series)
#     b : array_like
#         Cayley-Klein parameter beta (final value or time series)

#     Notes
#     -----
#     - B1/B0 are in mT
#     - G in mT/m
#     - pos in m
#     - Nt = number of timepoints
#     - Ns = number of spatial positions
#     - Nc = number of coils
#     """
#     # Determine which module to use based on input types
#     module = np
#     for x in [B1, G, pos, sens, B0]:
#         if _is_torch_tensor(x):
#             import torch

#             module = torch
#             break

#     # Constants
#     gam = 267522.1199722082  # radians per sec per mT

#     # Default parameters
#     dt = kwargs.get("dt", 6.4e-6)

#     # Handle M0 initialization
#     if "M0" in kwargs:
#         M0 = kwargs["M0"]
#         # Convert to appropriate module if needed
#         if _is_torch_tensor(M0) and module.__name__ == "numpy":
#             M0 = np.array(M0.detach().cpu().numpy())
#         elif not _is_torch_tensor(M0) and module.__name__ == "torch":
#             M0 = module.tensor(M0)
#     else:
#         if module.__name__ == "torch":
#             M0 = module.tensor([0.0, 0.0, 1.0])
#         else:
#             M0 = np.array([0.0, 0.0, 1.0])

#     # Get dimensions
#     Ns = pos.shape[0]  # Number of spatial positions
#     Nt = G.shape[0]  # Number of time points

#     # Initialize state variables
#     statea = _ones(Ns, dtype=B1.dtype if hasattr(B1, "dtype") else complex, module=module)
#     stateb = _zeros(Ns, dtype=B1.dtype if hasattr(B1, "dtype") else complex, module=module)

#     # Sum up RF over coils: bxy = sens * B1.'
#     bxy = module.matmul(sens, _transpose(B1))
#     # Sum up gradient over channels: bz = pos * G'
#     bz = module.matmul(pos, _transpose(G))

#     # Add off-resonance
#     if len(B0.shape) == 1:
#         B0 = B0.reshape(-1, 1)
#     bz = bz + module.tile(B0, (1, Nt))

#     # Determine if we need to return all states
#     returnallstate = kwargs.get("returnallstate", False)

#     if returnallstate:
#         a = _zeros((Ns, Nt), dtype=B1.dtype if hasattr(B1, "dtype") else complex, module=module)
#         b = _zeros((Ns, Nt), dtype=B1.dtype if hasattr(B1, "dtype") else complex, module=module)

#     # Compute these out of loop
#     # Phi = dt*gam*(abs(bxy).^2+bz.^2).^0.5
#     Phi = dt * gam * _sqrt(_abs(bxy) ** 2 + bz**2, module=module)

#     # Normfact = dt*gam*(Phi.^(-1))
#     Normfact = dt * gam / Phi
#     # Handle division by zero
#     Normfact = _where(_isfinite(Normfact), Normfact, _zeros_like(Normfact, module=module), module=module)

#     # Loop over time
#     for tt in range(Nt):
#         phi = -Phi[:, tt]  # sign reverse to define clockwise rotation
#         normfact = Normfact[:, tt]

#         nxy = normfact * bxy[:, tt]
#         nxy = _where(_isfinite(nxy), nxy, _zeros_like(nxy, module=module), module=module)

#         nz = normfact * bz[:, tt]
#         nz = _where(_isfinite(nz), nz, _zeros_like(nz, module=module), module=module)

#         cp = _cos(phi / 2, module=module)
#         sp = _sin(phi / 2, module=module)

#         alpha = cp - 1j * nz * sp
#         beta = -1j * nxy * sp

#         tmpa = alpha * statea - _conj(beta) * stateb
#         tmpb = beta * statea + _conj(alpha) * stateb

#         statea = tmpa
#         stateb = tmpb

#         if returnallstate:
#             a[:, tt] = statea
#             b[:, tt] = stateb

#     # Return final alpha, beta if not returning the whole state progression
#     if not returnallstate:
#         a = statea
#         b = stateb

#     # Calculate final magnetization state (M0 can be 3x1 or 3xNs)
#     if len(M0.shape) == 1:
#         # M0 is [x, y, z]
#         mxy0 = _complex(M0[0], M0[1], module=module)
#         mz0 = M0[2]

#         # Broadcast to match Ns
#         mxy0 = module.tile(mxy0, (Ns,))
#         mz0 = module.tile(mz0, (Ns,))
#     else:
#         # M0 is 3xNs or Nsx3
#         if M0.shape[0] == 3:
#             # M0 is 3xNs
#             mxy0 = _complex(M0[0], M0[1], module=module)
#             mz0 = M0[2]
#         else:
#             # M0 is Nsx3
#             mxy0 = _complex(M0[:, 0], M0[:, 1], module=module)
#             mz0 = M0[:, 2]

#     # Calculate final magnetization
#     mxy = 2 * mz0 * _conj(statea) * stateb + mxy0 * _conj(statea) ** 2 - _conj(mxy0) * stateb**2
#     mz = mz0 * (statea * _conj(statea) - stateb * _conj(stateb))
#     mz = mz + 2 * _real(mxy0 * _conj(statea) * (-_conj(stateb)), module=module)

#     # If returning all, then convert to Mxy(t) Mz(t)
#     if returnallstate:
#         if len(M0.shape) == 1:
#             # M0 is [x, y, z]
#             mxy0 = _complex(M0[0], M0[1], module=module)
#             mz0 = M0[2]

#             # Broadcast to match Ns
#             mxy0 = module.tile(mxy0, (Ns,))
#             mz0 = module.tile(mz0, (Ns,))
#         else:
#             # M0 is 3xNs or Nsx3
#             if M0.shape[0] == 3:
#                 # M0 is 3xNs
#                 mxy0 = _complex(M0[0], M0[1], module=module)
#                 mz0 = M0[2]
#             else:
#                 # M0 is Nsx3
#                 mxy0 = _complex(M0[:, 0], M0[:, 1], module=module)
#                 mz0 = M0[:, 2]

#         mxyt = 2 * mz0.reshape(-1, 1) * _conj(a) * b
#         mxyt = mxyt + mxy0.reshape(-1, 1) * _conj(a) ** 2
#         mxyt = mxyt - _conj(mxy0).reshape(-1, 1) * b**2

#         mzt = mz0.reshape(-1, 1) * (a * _conj(a) - b * _conj(b))
#         mzt = mzt + 2 * _real(mxy0.reshape(-1, 1) * _conj(a) * (-_conj(b)), module=module)

#         return mxy, module.real(mz), mxyt, module.real(mzt), a, b

#     return mxy, module.real(mz), a, b

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
    Normfact = dt * gam / Phi
    Normfact[~torch.isfinite(Normfact)] = 0.0  # Handle division by zero

    # Loop over time
    for tt in range(Nt):
        phi = -Phi[:, tt]  # sign reverse to define clockwise rotation

        cp = torch.cos(phi / 2)
        sp = torch.sin(phi / 2)

        alpha = cp - 1j*bz[:, tt]*gam*dt*(0.5 - phi**2/48)
        beta = -1j*bxy[:, tt]*gam*dt*(0.5 - phi**2/48)

        # print('='*100)
        # print(f'alpha: {alpha}')
        # print(f'alphaNew: {alphaNew}')
        # print(f'alpha diff: {torch.max(torch.abs(alpha-alphaNew))}')
        # print('-'*100)
        # print(f'beta: {beta}')
        # print(f'betaNew: {betaNew}')
        # print(f'beta diff: {torch.max(torch.abs(beta-betaNew))}')
        # print('='*100)

        tmpa = alpha * statea - torch.conj(beta) * stateb
        tmpb = beta * statea + torch.conj(alpha) * stateb

        statea = tmpa
        stateb = tmpb

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
