import numpy as np
import torch
from utils_bloch.simulation_utils import *


def blochsim_CK_timeprofile(B1, G, pos, sens, B0_list, M0, dt_num, **kwargs):
    Nb, Ns, Nt, bxy, bz = setup_simulation(G, pos, sens, B0_list, B1)

    alpha, beta = compute_alpha_beta(bxy, bz, dt_num)
    statea, stateb = time_loop_complex_timeprof(alpha, beta, Nb, Ns, Nt, B1.device)
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
