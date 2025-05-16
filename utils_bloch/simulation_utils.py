import torch
import numpy as np
import torch


def setup_simulation(G, pos, sens, B0_list, B1):
    from params import gamma

    # Constants
    G = torch.column_stack((0 * G.flatten(), 0 * G.flatten(), G.flatten()))

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
    return gamma, Nb, Ns, Nt, bxy, bz


def my_sinc(x):
    if x.device != torch.device("mps:0"):
        return torch.sinc(x / np.pi)
    large_x = torch.abs(x) > 1e-2
    result = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    result[~large_x] = 1 - x[~large_x] ** 2 / 6 + x[~large_x] ** 4 / 120 + x[~large_x] ** 6 / 5040
    result[large_x] = torch.sin(x[large_x]) / x[large_x]
    return result


def sinc_of_root(small_x):
    return 1 - small_x / 6 + small_x**2 / 120 + small_x**3 / 5040


def cos_of_root(small_x):
    return 1 - small_x / 2 + small_x**2 / 24 - small_x**3 / 720 + small_x**4 / 40320


def compute_alpha_beta(bxy, bz, dt, gam, B1):
    normSquared = (bxy * torch.conj(bxy)).real + bz**2  # Nb x Ns x Nt
    posNormPts = normSquared > 0
    Phi = torch.zeros(normSquared.shape, dtype=torch.float32, device=B1.device, requires_grad=False)
    Phi[posNormPts] = dt * gam * torch.sqrt(normSquared[posNormPts])
    sinc_part = -1j * gam * dt * 0.5 * torch.sinc(Phi / 2 / torch.pi)
    alpha = torch.cos(Phi / 2) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt
    return alpha, beta.squeeze()


def compute_alpha_beta_without_sqrt(bxy, bz, dt, gam, B1):
    Phi_half_squared = (0.5 * dt * gam) ** 2 * (bxy * torch.conj(bxy)).real + bz**2  # Nb x Ns x Nt
    sinc_part = -1j * gam * dt * 0.5 * sinc_of_root(Phi_half_squared)
    alpha = cos_of_root(Phi_half_squared) - bz * sinc_part
    beta = -bxy.unsqueeze(0) * sinc_part  # Nb x Ns x Nt
    return alpha, beta


def time_loop_complex_timeprof(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    Nb: int,
    Ns: int,
    Nt: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    statea = torch.zeros((Nb, Ns, Nt), dtype=torch.complex64, device=device)
    stateb = torch.zeros((Nb, Ns, Nt), dtype=torch.complex64, device=device)
    statea[:, :, 0] = 1.0
    for tt in range(Nt - 1):
        alpha_t = alpha[:, :, tt]
        beta_t = beta[:, :, tt]
        statea_t = statea[:, :, tt]
        stateb_t = stateb[:, :, tt]
        statea[:, :, tt + 1] = alpha_t * statea_t - torch.conj(beta_t) * stateb_t
        stateb[:, :, tt + 1] = beta_t * statea_t + torch.conj(alpha_t) * stateb_t
    return (statea, stateb)


def time_loop_complex(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    Nb: int,
    Ns: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    statea = torch.ones((Nb, Ns), dtype=torch.complex64, device=device)
    stateb = torch.zeros((Nb, Ns), dtype=torch.complex64, device=device)

    for tt in range(alpha.shape[-1]):
        alpha_t = alpha[:, :, tt]
        beta_t = beta[:, :, tt]
        tmpa = alpha_t * statea - torch.conj(beta_t) * stateb
        stateb = beta_t * statea + torch.conj(alpha_t) * stateb
        statea = tmpa
    return (statea, stateb)


@torch.jit.script
def time_loop_real(
    reAlpha: torch.Tensor,
    reBeta: torch.Tensor,
    imAlpha: torch.Tensor,
    imBeta: torch.Tensor,
    Nb: int,
    Ns: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reStatea = torch.ones((Nb, Ns), dtype=torch.float32, device=device)
    reStateb = torch.zeros((Nb, Ns), dtype=torch.float32, device=device)
    imStatea = torch.zeros((Nb, Ns), dtype=torch.float32, device=device)
    imStateb = torch.zeros((Nb, Ns), dtype=torch.float32, device=device)
    for tt in range(reAlpha.shape[-1]):
        reAlpha_t = reAlpha[:, :, tt]
        imAlpha_t = imAlpha[:, :, tt]
        reBeta_t = reBeta[:, :, tt]
        imBeta_t = imBeta[:, :, tt]
        retmpa = reAlpha_t * reStatea - imAlpha_t * imStatea - (reBeta_t * reStateb + imBeta_t * imStateb)
        imtmpa = reAlpha_t * imStatea + imAlpha_t * reStatea - (reBeta_t * imStateb - imBeta_t * reStateb)
        retmpb = reBeta_t * reStatea - imBeta_t * imStatea + (reAlpha_t * reStateb + imAlpha_t * imStateb)
        imStateb = reBeta_t * imStatea + imBeta_t * reStatea + (reAlpha_t * imStateb - imAlpha_t * reStateb)
        reStatea = retmpa
        imStatea = imtmpa
        reStateb = retmpb
    return (reStatea, imStatea, reStateb, imStateb)
