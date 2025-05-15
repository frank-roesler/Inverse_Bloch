from ..utils_bloch.blochsim_CK_freqprof import *


def plot_off_resonance(rf, grad, pos, sens, dt, B0, M0, freq_offsets_Hz):
    from params import flip_angle

    npts = len(freq_offsets_Hz)
    [mxy_profile, mz_profile] = blochsim_CK_freqprof(rf, grad, pos=pos, sens=sens, B0=B0, M0=M0, dt=dt, freq_offsets_Hz=freq_offsets_Hz)
    mxy_profile = mxy_profile.detach().cpu().numpy() if isinstance(mxy_profile, torch.Tensor) else mxy_profile
    mz_profile = mz_profile.detach().cpu().numpy() if isinstance(mz_profile, torch.Tensor) else mz_profile
    pos = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else pos

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4))

    # Subplot 1: abs(Mxy)
    im1 = axes[0].imshow(
        np.abs(mxy_profile),
        extent=[
            linspace(-8000, 8000, npts).min() / 297.3 + 4.7,
            linspace(-8000, 8000, npts).max() / 297.3 + 4.7,
            squeeze(pos[:, 2] * 100).max(),
            squeeze(pos[:, 2] * 100).min(),
        ],
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
        unwrap(angle(mxy_profile), axis=0),
        extent=[
            linspace(-8000, 8000, npts).min() / 297.3 + 4.7,
            linspace(-8000, 8000, npts).max() / 297.3 + 4.7,
            squeeze(pos[:, 2] * 100).max(),
            squeeze(pos[:, 2] * 100).min(),
        ],
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
        extent=[
            linspace(-8000, 8000, npts).min() / 297.3 + 4.7,
            linspace(-8000, 8000, npts).max() / 297.3 + 4.7,
            squeeze(pos[:, 2] * 100).max(),
            squeeze(pos[:, 2] * 100).min(),
        ],
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

    # Subplot 4: abs(Mxy) zoomed in
    extent_full = [
        linspace(-8000, 8000, npts).min() / 297.3 + 4.7,
        linspace(-8000, 8000, npts).max() / 297.3 + 4.7,
        squeeze(pos[:, 2] * 100).max(),
        squeeze(pos[:, 2] * 100).min(),
    ]

    # # Calculate zoomed-in extent (centered with zoom factor of 2)
    # zoom_factor = 2
    # x_center = (extent_full[0] + extent_full[1]) / 2
    # x_range = (extent_full[1] - extent_full[0]) / zoom_factor
    # y_center = (extent_full[2] + extent_full[3]) / 2
    # y_range = (extent_full[2] - extent_full[3]) / zoom_factor
    # extent_zoomed = [x_center - x_range / 2, x_center + x_range / 2, y_center + y_range / 2, y_center - y_range / 2]

    # # Extract the zoomed-in data
    # x_indices = np.logical_and(
    #     linspace(-8000, 8000, npts) / 297.3 + 4.7 >= extent_zoomed[0],
    #     linspace(-8000, 8000, npts) / 297.3 + 4.7 <= extent_zoomed[1],
    # )
    # y_indices = np.logical_and(
    #     squeeze(pos[:, 2] * 100) >= extent_zoomed[3], squeeze(pos[:, 2] * 100) <= extent_zoomed[2]
    # )

    # mxy_zoomed = np.abs(mxy_profile[np.ix_(y_indices, x_indices)])

    # # Plot the zoomed-in data
    # im4 = axes[1, 0].imshow(mxy_zoomed, extent=extent_zoomed, aspect="auto")
    # axes[1, 0].set_xlabel("Off Resonance [ppm]")
    # axes[1, 0].set_ylabel("Spatial Pos [cm]")
    # axes[1, 0].axvline(4.7, color="r", linewidth=1.5)
    # axes[1, 0].set_title("abs(Mxy) (Zoomed In)")
    # axes[1, 0].invert_xaxis()
    # fig.colorbar(im4, ax=axes[1, 0])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
