from utils_bloch.blochsim_freqprof import *
import matplotlib.pyplot as plt
from torch import linspace, squeeze, angle
import os
from skimage.restoration import unwrap_phase
from constants import gamma, gam_hz_mt, larmor_mhz, water_ppm


def plot_off_resonance(
    rf,
    grad,
    fixed_inputs,
    freq_offsets_Hz,
    flip_angle,
    path=None,
):
    npts = len(freq_offsets_Hz)
    pos = fixed_inputs["pos"].detach().cpu()
    [mxy_profile, mz_profile] = blochsim_CK_freqprof(
        rf.detach().cpu(),
        grad.detach().cpu(),
        pos=pos,
        sens=fixed_inputs["sens"].detach().cpu(),
        B0=fixed_inputs["B0"].detach().cpu(),
        M0=fixed_inputs["M0"].detach().cpu(),
        dt=fixed_inputs["dt_num"],
        freq_offsets_Hz=freq_offsets_Hz,
    )

    mxy_profile = mxy_profile.detach().cpu()
    mz_profile = mz_profile.detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4))
    img_extent = [
        linspace(-8000, 8000, npts).min() / larmor_mhz + water_ppm,
        linspace(-8000, 8000, npts).max() / larmor_mhz + water_ppm,
        squeeze(pos * 100).max(),
        squeeze(pos * 100).min(),
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
    axes[0].axvline(water_ppm, color="r", linewidth=1.5)
    axes[0].set_title("abs(Mxy)")
    axes[0].invert_xaxis()
    fig.colorbar(im1, ax=axes[0])

    # Subplot 2: angle(Mxy)
    # phase = unwrap_phase(np.angle(mxy_profile))
    phase = np.angle(mxy_profile)

    im2 = axes[1].imshow(
        phase,
        extent=img_extent,
        aspect="auto",
    )
    axes[1].set_xlabel("Off Resonance [ppm]")
    axes[1].set_ylabel("Spatial Pos [cm]")
    axes[1].axvline(water_ppm, color="r", linewidth=1.5)
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
    axes[2].axvline(water_ppm, color="r", linewidth=1.5)
    axes[2].set_title("Mz")
    axes[2].invert_xaxis()
    fig.colorbar(im3, ax=axes[2])

    # Adjust layout and show the plot
    plt.tight_layout()
    if path is not None:
        np.save(os.path.join(os.path.dirname(path), "phase.npy"), phase)
        fig.savefig(os.path.join(os.path.dirname(path), "freqprof.png"), dpi=300)


def plot_some_b0_values(n_values, fixed_inputs, G, B1, tconfig, bconfig, path=None):
    from utils_train.utils import move_to
    from utils_infer import compute_phase_offsets
    from utils_bloch.setup import get_smooth_targets

    bconfig.n_b0_values = n_values
    target_z, target_xy, slice_centers_allB0, half_width = get_smooth_targets(tconfig, bconfig, function=torch.sigmoid, override_inputs=fixed_inputs)

    t_B1 = fixed_inputs["t_B1"]
    pos = fixed_inputs["pos"]
    freq_offsets_Hz = torch.linspace(-larmor_mhz * water_ppm, 0.0, n_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(fixed_inputs["B0"] + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=fixed_inputs["sens"], B0_list=B0, M0=fixed_inputs["M0"], dt=fixed_inputs["dt_num"])
    (mxy, mz, pos, target_xy, target_z, t_B1, G, B1) = move_to((mxy, mz, pos, target_xy, target_z, t_B1, G, B1), torch.device("cpu"))
    delta_t = np.diff(t_B1, axis=0)
    for j, ff in enumerate(range(len(freq_offsets_Hz))):
        fig, ax = plt.subplots(2, 2, figsize=(14, 6))
        fig.suptitle(f"B0: {-freq_offsets_Hz[ff]/larmor_mhz:.1f} ppm")
        mxy_abs = np.abs(mxy[ff, :])
        mxy_argmax = np.argmax(mxy_abs)
        flip_angle = np.arctan(mxy_abs[mxy_argmax] / mz[ff, mxy_argmax]) / 2 / np.pi * 360
        print(f"Flip angle for B0 {ff}: {flip_angle:.2f} degrees")
        ax[0, 0].plot(t_B1, np.real(B1), linewidth=0.8)
        ax[0, 0].plot(t_B1, np.imag(B1), linewidth=0.8)
        ax[0, 0].plot(t_B1, np.abs(B1), linewidth=0.8, linestyle="dotted")
        ax[0, 1].plot(t_B1, G * np.ones(t_B1.shape), linewidth=0.8)
        ax01 = ax[0, 1].twinx()
        ax01.plot(t_B1[:-1], np.diff(G, axis=0) / delta_t, linewidth=1)
        ax[1, 0].plot(pos, np.real(mz[ff, :]), linewidth=0.8)
        ax[1, 0].plot(pos, target_z[ff, :], linewidth=0.8)
        ax[1, 1].plot(pos, mxy_abs, linewidth=0.8)
        ax[1, 1].plot(pos, target_xy[ff, :], linewidth=0.8)

        # phase = np.unwrap(np.angle(mxy[ff, :]))
        phase = np.angle(mxy[ff, :])
        phasemin = np.inf
        phasemax = -np.inf
        slice_centers_current_b0 = slice_centers_allB0[ff]
        for i, c in enumerate(slice_centers_current_b0):
            phase_loc = phase[c - half_width : c + half_width]
            phasemin = np.min([phasemin, np.min(phase_loc)])
            phasemax = np.max([phasemax, np.max(phase_loc)])
        phasemin -= 0.5 * np.abs(phasemax - phasemin)
        phasemax += 0.5 * np.abs(phasemax - phasemin)

        phase[target_xy[ff, :] < 0.5] = np.nan
        ax_phase = ax[1, 1].twinx()
        ax_phase.set_ylabel("Phase (radians)")
        ax_phase.set_ylim(phasemin, phasemax)
        ax_phase.plot(pos, phase, linewidth=0.8, color="g")
        if path is not None:
            fig.savefig(os.path.join(os.path.dirname(path), f"B0_{ff}.png"), dpi=300)
