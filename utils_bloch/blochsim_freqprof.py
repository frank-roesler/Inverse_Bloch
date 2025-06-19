import numpy as np
from utils_bloch.blochsim_batch import blochsim_CK_batch
import torch
from utils_bloch.simulation_utils import time_loop_complex, compute_alpha_beta
import params


def blochsim_CK_freqprof(fixed_inputs, B1, G, pos, sens, B0, M0=np.array([0, 0, 1]), dt=6.4e-6, freq_offsets_Hz=np.array([0]), **kwargs):
    gamma = fixed_inputs["gam"]
    gamma_hz_mt = fixed_inputs["gam_hz_mt"]
    pos = torch.stack([torch.zeros_like(pos), torch.zeros_like(pos), pos], dim=-1)

    G = torch.column_stack((0 * G.flatten(), 0 * G.flatten(), G.flatten()))
    Ns = pos.shape[0]
    Nt = G.shape[0]
    Nfreq = len(freq_offsets_Hz)
    bxy = torch.matmul(sens, B1.T)
    bz_grad_component = torch.matmul(pos, G.T)

    B0_base = B0.flatten()
    if M0.ndim == 1:
        mxy0_base = M0[0] + 1j * M0[1]
        mz0_base = M0[2]
    else:
        mxy0_base = M0[0, :] + 1j * M0[1, :]
        mz0_base = M0[2, :]
        mxy0_base = mxy0_base.flatten()
        mz0_base = mz0_base.flatten()

    B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
    mxy_profile = torch.zeros((Ns, Nfreq), dtype=complex)
    mz_profile = torch.zeros((Ns, Nfreq))

    print(f"Starting simulation for {Nfreq} frequency offsets...")
    for ff in range(Nfreq):
        print(f"  Simulating frequency offset {ff + 1} / {Nfreq} ({freq_offsets_Hz[ff]:.2f} Hz)")

        B0_total_this_freq = B0_base + B0_freq_offsets_mT[ff]
        bz = bz_grad_component + torch.tile(B0_total_this_freq, (Nt, 1)).T
        alpha, beta = compute_alpha_beta(bxy, bz, dt, gamma)
        beta = beta.squeeze()
        statea, stateb = time_loop_complex(alpha.unsqueeze(0), beta.unsqueeze(0), 1, Ns, B1.device)
        statea, stateb = statea.squeeze(), stateb.squeeze()
        stateaBar, statebBar = torch.conj(statea), torch.conj(stateb)

        mxy_profile[:, ff] = 2 * mz0_base * stateaBar * stateb + mxy0_base * stateaBar**2 - torch.conj(mxy0_base) * stateb**2
        mz_profile[:, ff] = mz0_base * (statea * stateaBar).real - (stateb * statebBar).real - 2 * (mxy0_base * stateaBar * statebBar).real

    print("Frequency profile simulation complete.")

    return mxy_profile, mz_profile


def blochsim_CK_freqprof_batch(B1, G, pos, sens, B0, freq_offsets_Hz, dt, M0=torch.Tensor([0, 0, 1]), **kwargs):
    from params import gamma, gamma_hz_mt

    B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(B0 + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=sens, B0_list=B0, M0=M0, dt=dt)
    return mxy.permute(1, 0), mz.permute(1, 0)
