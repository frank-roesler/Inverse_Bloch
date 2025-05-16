from utils_bloch.blochsim_freqprof import *
import matplotlib.pyplot as plt
from torch import linspace, squeeze, angle
from utils_train.utils import torch_unwrap, move_to


def plot_off_resonance(rf, grad, pos, sens, dt, B0, M0, freq_offsets_Hz):
    from params import flip_angle

    npts = len(freq_offsets_Hz)
    [mxy_profile, mz_profile] = blochsim_CK_freqprof(rf, grad, pos=pos, sens=sens, B0=B0, M0=M0, dt=dt, freq_offsets_Hz=freq_offsets_Hz)

    (mxy_profile, mz_profile, pos) = move_to((mxy_profile, mz_profile, pos), torch.device("cpu"))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4))
    img_extent = [
        linspace(-8000, 8000, npts).min() / 297.3 + 4.7,
        linspace(-8000, 8000, npts).max() / 297.3 + 4.7,
        squeeze(pos[:, 2] * 100).max(),
        squeeze(pos[:, 2] * 100).min(),
    ]
    # Subplot 1: abs(Mxy)
    im1 = axes[0].imshow(
        np.abs(mxy_profile),
        extent=img_extent,
        aspect="auto",
        vmin=0,
        vmax=np.sin(flip_angle),
    )
    axes[0].set_xlabel("Off Resonance [ppm]")
    axes[0].set_ylabel("Spatial Pos [cm]")
    axes[0].axvline(4.7, color="r", linewidth=1.5)
    axes[0].set_title("abs(Mxy)")
    axes[0].invert_xaxis()
    fig.colorbar(im1, ax=axes[0])

    # Subplot 2: angle(Mxy)
    im2 = axes[1].imshow(
        np.unwrap(angle(mxy_profile), axis=0),
        extent=img_extent,
        aspect="auto",
    )
    axes[1].set_xlabel("Off Resonance [ppm]")
    axes[1].set_ylabel("Spatial Pos [cm]")
    axes[1].axvline(4.7, color="r", linewidth=1.5)
    axes[1].set_title("angle(Mxy)")
    axes[1].invert_xaxis()
    fig.colorbar(im2, ax=axes[1])

    # Subplot 3: abs(Mz)
    im3 = axes[2].imshow(
        np.abs(mz_profile),
        extent=img_extent,
        aspect="auto",
        vmax=1,
        vmin=np.cos(flip_angle),
    )
    axes[2].set_xlabel("Off Resonance [ppm]")
    axes[2].set_ylabel("Spatial Pos [cm]")
    axes[2].axvline(4.7, color="r", linewidth=1.5)
    axes[2].set_title("Mz")
    axes[2].invert_xaxis()
    fig.colorbar(im3, ax=axes[2])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_some_b0_values(n_values, pos, sens, G, B1, B0, M0, target_xy, target_z, t_B1, dt):
    from params import gamma_hz_mt

    freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, n_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(B0 + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=sens, B0_list=B0, M0=M0, dt=dt)
    (mxy, mz, pos, target_xy, target_z, t_B1, G, B1) = move_to((mxy, mz, pos, target_xy, target_z, t_B1, G, B1), torch.device("cpu"))
    delta_t = np.diff(t_B1, axis=0)
    for ff in range(len(freq_offsets_Hz)):
        fig, ax = plt.subplots(2, 2, figsize=(14, 6))
        ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
        ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
        ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
        ax[0, 1].plot(t_B1, G * np.ones(t_B1.shape), linewidth=0.8)
        ax01 = ax[0, 1].twinx()
        ax01.plot([], [])
        ax01.plot(t_B1[:-1], np.diff(G, axis=0) / delta_t, linewidth=1)
        ax[1, 0].plot(pos[:, 2], np.real(mz[ff, :]), linewidth=0.8)
        ax[1, 0].plot(pos[:, 2], target_z, linewidth=0.8)
        ax[1, 1].plot(pos[:, 2], np.abs(mxy[ff, :]), linewidth=0.8)
        ax[1, 1].plot(pos[:, 2], target_xy, linewidth=0.8)

        phase = np.unwrap(np.angle(mxy[ff, :]))
        phasemin = np.min(phase)
        phasemax = np.max(phase)
        phase[target_xy < 0.5] = np.nan
        ax_phase = ax[1, 1].twinx()
        ax_phase.set_ylabel("Phase (radians)")
        ax_phase.set_ylim(phasemin, phasemax)
        ax_phase.plot(pos[:, 2], phase, linewidth=0.8, color="g")
        # plt.savefig("forward.png", dpi=300)
        plt.show()
