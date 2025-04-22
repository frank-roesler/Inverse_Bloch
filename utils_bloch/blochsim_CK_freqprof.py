import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, squeeze, angle, unwrap

G3 = lambda Gz: np.column_stack((0 * Gz.flatten(), 0 * Gz.flatten(), Gz.flatten()))


def blochsim_CK_freqprof(B1, G, pos, sens, B0, **kwargs):
    gam = 267522.1199722082
    gam_hz_mt = gam / (2 * np.pi)
    dt = 6.4e-6
    M0 = np.array([0, 0, 1])
    freq_offsets_Hz = np.array([0])

    # Parse optional arguments
    if "dt" in kwargs:
        dt = kwargs["dt"]
    if "M0" in kwargs:
        M0 = np.array(kwargs["M0"])
    if "freq_offsets_Hz" in kwargs:
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


def plot_off_resonance(rf, grad, pos, Nz, dt):
    [mxy_profile, mz_profile] = blochsim_CK_freqprof(
        rf, G3(grad), pos, np.ones((Nz, 1)), np.zeros((Nz, 1)), dt=dt, freq_offsets_Hz=linspace(-8000, 8000, 512)
    )

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Subplot 1: abs(Mxy)
    im1 = axes[0].imshow(
        np.abs(mxy_profile),
        extent=[
            linspace(-8000, 8000, 512).min() / 297.3 + 4.7,
            linspace(-8000, 8000, 512).max() / 297.3 + 4.7,
            squeeze(pos[:, 2] * 100).max(),
            squeeze(pos[:, 2] * 100).min(),
        ],
        aspect="auto",
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
            linspace(-8000, 8000, 512).min() / 297.3 + 4.7,
            linspace(-8000, 8000, 512).max() / 297.3 + 4.7,
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
            linspace(-8000, 8000, 512).min() / 297.3 + 4.7,
            linspace(-8000, 8000, 512).max() / 297.3 + 4.7,
            squeeze(pos[:, 2] * 100).max(),
            squeeze(pos[:, 2] * 100).min(),
        ],
        aspect="auto",
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
