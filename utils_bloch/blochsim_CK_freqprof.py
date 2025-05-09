import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, squeeze, angle, unwrap
from utils_bloch.blochsim_batch import blochsim_CK_batch
import torch

G3 = lambda Gz: np.column_stack((0 * Gz.flatten(), 0 * Gz.flatten(), Gz.flatten()))


def blochsim_CK_freqprof(B1, G, pos, sens, B0, **kwargs):
    G = G3(G.cpu().numpy())
    B1 = B1.cpu().numpy()
    pos = pos.cpu().numpy()
    sens = sens.cpu().numpy()
    B0 = B0.cpu().numpy()
    gam = 267522.1199722082
    gam_hz_mt = gam / (2 * np.pi)
    dt = 6.4e-6
    M0 = np.array([0, 0, 1])
    freq_offsets_Hz = np.array([0])

    # Parse optional arguments
    if "dt" in kwargs:
        dt = kwargs["dt"]
    if "M0" in kwargs:
        if isinstance(kwargs["M0"], torch.Tensor):
            M0 = kwargs["M0"].cpu().numpy()
        else:
            M0 = np.array(kwargs["M0"])

    if "freq_offsets_Hz" in kwargs:
        if isinstance(kwargs["freq_offsets_Hz"], torch.Tensor):
            freq_offsets_Hz = kwargs["freq_offsets_Hz"].cpu().numpy()
        else:
            freq_offsets_Hz = np.array(kwargs["freq_offsets_Hz"])

    Ns = pos.shape[0]
    Nt = G.shape[0]
    Nfreq = len(freq_offsets_Hz)

    B0_base = B0.flatten()
    if M0.ndim == 1:
        mxy0_base = M0[0] + 1j * M0[1]
        mz0_base = M0[2]
    else:
        mxy0_base = M0[0, :] + 1j * M0[1, :]
        mz0_base = M0[2, :]
        mxy0_base = mxy0_base.flatten()
        mz0_base = mz0_base.flatten()

    B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
    bxy = sens * B1.T
    bz_grad_component = np.dot(pos, G.T)
    mxy_profile = np.zeros((Ns, Nfreq), dtype=complex)
    mz_profile = np.zeros((Ns, Nfreq))

    print(f"Starting simulation for {Nfreq} frequency offsets...")
    for ff in range(Nfreq):
        print(f"  Simulating frequency offset {ff + 1} / {Nfreq} ({freq_offsets_Hz[ff]:.2f} Hz)")

        B0_total_this_freq = B0_base + B0_freq_offsets_mT[ff]

        print(np.max(B0_total_this_freq).item())

        bz = bz_grad_component + np.tile(B0_total_this_freq, (Nt, 1)).T

        statea = np.ones((Ns,), dtype=complex)
        stateb = np.zeros((Ns,), dtype=complex)

        Phi = dt * gam * np.sqrt(np.abs(bxy) ** 2 + bz**2)
        sinc_part = -1j * gam * dt * 0.5 * np.sinc(Phi / 2 / np.pi)
        alpha = np.cos(Phi / 2) - bz * sinc_part
        beta = -bxy * sinc_part
        alphaBar = np.conj(alpha)
        betaBar = np.conj(beta)

        # Loop over time
        for tt in range(Nt):
            tmpa = alpha[:, tt] * statea - betaBar[:, tt] * stateb
            stateb = beta[:, tt] * statea + alphaBar[:, tt] * stateb
            statea = tmpa

        mxy_this_freq = (
            2 * mz0_base * np.conj(statea) * stateb + mxy0_base * np.conj(statea) ** 2 - np.conj(mxy0_base) * stateb**2
        )
        mz_val1 = mz0_base * (statea * np.conj(statea) - stateb * np.conj(stateb))
        mz_val2 = 2 * np.real(mxy0_base * np.conj(statea) * (-np.conj(stateb)))
        mz_this_freq = mz_val1 + mz_val2

        mxy_profile[:, ff] = mxy_this_freq
        mz_profile[:, ff] = mz_this_freq

    print("Frequency profile simulation complete.")

    return mxy_profile, mz_profile


def blochsim_CK_freqprof2(B1, G, pos, sens, B0, freq_offsets_Hz, dt, M0=torch.Tensor([0, 0, 1]), **kwargs):
    gam = 267522.1199722082
    gam_hz_mt = gam / (2 * np.pi)
    B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(B0 + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=sens, B0_list=B0, M0=M0, dt=dt)
    return mxy.permute(1, 0), mz.permute(1, 0)


def plot_off_resonance(rf, grad, pos, sens, dt, B0, M0, freq_offsets_Hz):
    from params import flip_angle

    npts = len(freq_offsets_Hz)
    [mxy_profile, mz_profile] = blochsim_CK_freqprof(
        rf, grad, pos=pos, sens=sens, B0=B0, M0=M0, dt=dt, freq_offsets_Hz=freq_offsets_Hz
    )
    mxy_profile = mxy_profile.detach().cpu().numpy() if isinstance(mxy_profile, torch.Tensor) else mxy_profile
    mz_profile = mz_profile.detach().cpu().numpy() if isinstance(mz_profile, torch.Tensor) else mz_profile
    pos = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else pos

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

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
        vmin=np.sin(flip_angle),
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
