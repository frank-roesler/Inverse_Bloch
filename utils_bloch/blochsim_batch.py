import numpy as np
import torch
from utils_bloch.blochsim_CK import my_sinc


# @torch.jit.script
# def time_loop(
#     reAlpha: torch.Tensor,
#     reBeta: torch.Tensor,
#     imAlpha: torch.Tensor,
#     imBeta: torch.Tensor,
#     Nb: int,
#     Ns: int,
#     Nt: int,
#     device: torch.device,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     reStatea = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
#     reStateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
#     imStatea = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
#     imStateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
#     reStatea[:, :, 0] = 1.0
#     for tt in range(Nt - 1):
#         reStatea_t = reStatea[:, :, tt].clone()
#         imStatea_t = imStatea[:, :, tt].clone()
#         reStateb_t = reStateb[:, :, tt].clone()
#         imStateb_t = imStateb[:, :, tt].clone()
#         reStatea[:, :, tt + 1] = reAlpha[:, :, tt] * reStatea_t - imAlpha[:, :, tt] * imStatea_t - (reBeta[:, :, tt] * reStateb_t + imBeta[:, :, tt] * imStateb_t)
#         imStatea[:, :, tt + 1] = reAlpha[:, :, tt] * imStatea_t + imAlpha[:, :, tt] * reStatea_t - (reBeta[:, :, tt] * imStateb_t - imBeta[:, :, tt] * reStateb_t)
#         reStateb[:, :, tt + 1] = reBeta[:, :, tt] * reStatea_t - imBeta[:, :, tt] * imStatea_t + (reAlpha[:, :, tt] * reStateb_t + imAlpha[:, :, tt] * imStateb_t)
#         imStateb[:, :, tt + 1] = reBeta[:, :, tt] * imStatea_t + imBeta[:, :, tt] * reStatea_t + (reAlpha[:, :, tt] * imStateb_t - imAlpha[:, :, tt] * reStateb_t)
#     return (reStatea, imStatea, reStateb, imStateb)


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
    reStatea = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    reStateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    imStatea = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    imStateb = torch.zeros((Nb, Ns, Nt), dtype=torch.float32, device=device)
    reStatea[:, :, 0] = 1.0

    for tt in range(Nt - 1):
        reStatea_t = reStatea[:, :, tt].clone()
        imStatea_t = imStatea[:, :, tt].clone()
        reStateb_t = reStateb[:, :, tt].clone()
        imStateb_t = imStateb[:, :, tt].clone()

        reAlpha_tt = reAlpha[:, :, tt]
        imAlpha_tt = imAlpha[:, :, tt]
        reBeta_tt = reBeta[:, :, tt]
        imBeta_tt = imBeta[:, :, tt]

        reStatea[:, :, tt + 1] = reAlpha_tt * reStatea_t - imAlpha_tt * imStatea_t - (reBeta_tt * reStateb_t + imBeta_tt * imStateb_t)
        imStatea[:, :, tt + 1] = reAlpha_tt * imStatea_t + imAlpha_tt * reStatea_t - (reBeta_tt * imStateb_t - imBeta_tt * reStateb_t)
        reStateb[:, :, tt + 1] = reBeta_tt * reStatea_t - imBeta_tt * imStatea_t + (reAlpha_tt * reStateb_t + imAlpha_tt * imStateb_t)
        imStateb[:, :, tt + 1] = reBeta_tt * imStatea_t + imBeta_tt * reStatea_t + (reAlpha_tt * imStateb_t - imAlpha_tt * reStateb_t)
    return reStatea, imStatea, reStateb, imStateb


def sinc_of_root(small_x):
    return 1 - small_x / 6 + small_x**2 / 120 + small_x**3 / 5040


def cos_of_root(small_x):
    return 1 - small_x / 2 + small_x**2 / 24 - small_x**3 / 720 + small_x**4 / 40320


def compute_alpha_beta(bxy, bz, dt, gam):
    normSquared = (bxy * torch.conj(bxy)).real + bz**2  # Nb x Ns x Nt
    posNormPts = normSquared > 0
    Phi = torch.zeros(normSquared.shape, dtype=torch.float32, device=bxy.device, requires_grad=False)
    Phi[posNormPts] = dt * gam * torch.sqrt(normSquared[posNormPts])
    sinc_part = -1j * gam * dt * 0.5 * torch.sinc(Phi / 2 / torch.pi)
    alpha = torch.cos(Phi / 2) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt
    return alpha, beta


def compute_alpha_beta_without_sqrt(bxy, bz, dt, gam):
    Phi_half_squared = (0.5 * dt * gam) ** 2 * ((bxy * torch.conj(bxy)).real + bz**2)  # Nb x Ns x Nt
    sinc_part = -1j * gam * dt * 0.5 * sinc_of_root(Phi_half_squared)
    alpha = cos_of_root(Phi_half_squared) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt
    return alpha, beta


def blochsim_CK_batch(B1, G, pos, sens, dx, B0_list, M0, slice_centers, slice_half_width, dt=6.4e-6):
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
    bz = bz + B0_list  # Final shape: (Nb, Ns, Nt)

    alpha, beta = compute_alpha_beta_without_sqrt(bxy, bz, dt, gam)

    # Loop over time
    reStatea, imStatea, reStateb, imStateb = time_loop(
        torch.real(alpha),
        torch.real(beta),
        torch.imag(alpha),
        torch.imag(beta),
        Nb,
        Ns,
        Nt,
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
    mz_batch = mz0 * ((statea * stateaBar).real - (stateb * statebBar).real) - 2 * (mxy0 * stateaBar * statebBar).real

    mxy_batch_integrated = []
    for c in slice_centers:
        mxy_batch_integrated.append(torch.mean(torch.abs(mxy_batch[:, c - slice_half_width : c + slice_half_width, :]), dim=1, keepdim=True))

    mxy_batch_integrated = torch.cat(mxy_batch_integrated, dim=1)

    return mxy_batch[:, :, -1], mz_batch.real[:, :, -1], mxy_batch_integrated
