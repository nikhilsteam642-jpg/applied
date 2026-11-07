"""
Scientific Visualization Module
Generates publication-quality dark navy & cyan themed plots for time-domain waveforms,
FFT spectra, pole-zero diagrams, filter frequency responses, and dual spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram
from typing import Dict, Any, List, Optional, Tuple

# Scientific Dashboard Color Palette
BG_DARK = "#0B132B"
CARD_BG = "#1C2541"
TEXT_MAIN = "#E0FBFC"
TEXT_MUTED = "#8D99AE"
CYAN_ACCENT = "#00B4D8"
TEAL_ACCENT = "#48CAE4"
EMERALD_ACCENT = "#06D6A0"
CORAL_ACCENT = "#EF476F"
AMBER_ACCENT = "#FFD166"
GRID_COLOR = "#2A385B"


def apply_theme(fig, axes):
    """Applies consistent dark navy scientific styling to matplotlib figures."""
    fig.patch.set_facecolor(BG_DARK)
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    elif isinstance(axes, list):
        axes_list = axes
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
        ax.xaxis.label.set_color(TEXT_MAIN)
        ax.yaxis.label.set_color(TEXT_MAIN)
        ax.title.set_color(TEXT_MAIN)


def plot_waveform_single(
    t: np.ndarray,
    x: np.ndarray,
    title: str = "Discrete-Time Signal x[n]",
    color: str = CYAN_ACCENT,
    max_display_points: int = 4000
) -> plt.Figure:
    """Plots a single discrete audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    apply_theme(fig, ax)

    # Downsample for snappy rendering if audio is long
    step = max(1, len(x) // max_display_points)
    t_plot = t[::step]
    x_plot = x[::step]

    ax.plot(t_plot, x_plot, color=color, linewidth=1.1, alpha=0.9, label="x[n]")
    ax.axhline(0, color=GRID_COLOR, linestyle="-", linewidth=1.0)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim([-1.05, 1.05])
    fig.tight_layout()
    return fig


def plot_waveform_comparison(
    t: np.ndarray,
    x_orig: np.ndarray,
    y_enh: np.ndarray,
    max_display_points: int = 4000
) -> plt.Figure:
    """
    Plots stacked waveforms: Original/Noisy Audio, Enhanced Audio, and Removed Residual Noise.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, dpi=100)
    apply_theme(fig, axes)

    N = min(len(x_orig), len(y_enh))
    step = max(1, N // max_display_points)
    t_plot = t[:N:step]
    x_o = x_orig[:N:step]
    y_e = y_enh[:N:step]
    residual = x_o - y_e

    # 1. Original / Noisy
    axes[0].plot(t_plot, x_o, color=CORAL_ACCENT, linewidth=1.1, alpha=0.9)
    axes[0].set_title("Input / Noisy Signal  x[n]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude", fontsize=9)
    axes[0].set_ylim([-1.1, 1.1])

    # 2. Enhanced
    axes[1].plot(t_plot, y_e, color=EMERALD_ACCENT, linewidth=1.1, alpha=0.9)
    axes[1].set_title("Enhanced Filtered Output  y[n]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Amplitude", fontsize=9)
    axes[1].set_ylim([-1.1, 1.1])

    # 3. Residual Removed Noise
    axes[2].plot(t_plot, residual, color=AMBER_ACCENT, linewidth=1.0, alpha=0.85)
    axes[2].set_title("Removed Noise Residual  e[n] = x[n] - y[n]", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (seconds)", fontsize=9)
    axes[2].set_ylabel("Amplitude", fontsize=9)
    axes[2].set_ylim([-1.1, 1.1])

    fig.tight_layout()
    return fig


def plot_fft_spectrum(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    dominant_peaks: Optional[List[Dict[str, float]]] = None,
    cutoff_freqs: Optional[List[float]] = None,
    title: str = "FFT Magnitude Spectrum |X[k]|"
) -> plt.Figure:
    """Plots single-sided FFT magnitude spectrum with peak markers."""
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=100)
    apply_theme(fig, ax)

    ax.plot(freqs, mag_db, color=CYAN_ACCENT, linewidth=1.2, label="Magnitude (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim([max(-90.0, float(np.min(mag_db) - 5)), 5.0])

    # Highlight dominant peaks
    if dominant_peaks:
        for p in dominant_peaks:
            f_pk = p["frequency_hz"]
            db_pk = p["magnitude_db_rel"]
            ax.plot(f_pk, db_pk, marker="o", markersize=6, color=AMBER_ACCENT)
            ax.annotate(
                f"{f_pk:.0f} Hz",
                xy=(f_pk, db_pk),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=AMBER_ACCENT,
                weight="bold"
            )

    # Cutoff indicators
    if cutoff_freqs:
        for fc in cutoff_freqs:
            if fc is not None and fc > 0:
                ax.axvline(fc, color=CORAL_ACCENT, linestyle="--", linewidth=1.5, label=f"Cutoff {fc:.0f} Hz")

    ax.legend(loc="upper right", facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_MAIN)
    fig.tight_layout()
    return fig


def plot_fft_comparison(
    freqs: np.ndarray,
    mag_orig_db: np.ndarray,
    mag_enh_db: np.ndarray,
    cutoff_freqs: Optional[List[float]] = None
) -> plt.Figure:
    """Overlays Original and Enhanced FFT spectra to demonstrate spectral attenuation."""
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    apply_theme(fig, ax)

    ax.plot(freqs, mag_orig_db, color=CORAL_ACCENT, linewidth=1.2, alpha=0.75, label="Original / Noisy Spectrum")
    ax.plot(freqs, mag_enh_db, color=EMERALD_ACCENT, linewidth=1.4, alpha=0.95, label="Enhanced Filtered Spectrum")

    if cutoff_freqs:
        for fc in cutoff_freqs:
            if fc is not None and fc > 0:
                ax.axvline(fc, color=AMBER_ACCENT, linestyle=":", linewidth=1.6, label=f"Filter Cutoff ({fc:.0f} Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Magnitude (dB)")
    ax.set_title("Spectral Comparison: Original vs Enhanced FFT Spectrum", fontsize=12, fontweight="bold")
    ax.set_ylim([-85.0, 5.0])
    ax.legend(loc="upper right", facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8.5, labelcolor=TEXT_MAIN)
    fig.tight_layout()
    return fig


def plot_pole_zero_diagram(
    zeros: np.ndarray,
    poles: np.ndarray,
    is_stable: bool,
    title: str = "Z-Domain Pole-Zero Constellation & Unit Circle"
) -> plt.Figure:
    """
    Plots the Z-domain complex plane with Unit Circle |z| = 1,
    Poles ('x'), Zeros ('o'), and Region of Convergence (ROC).
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=100)
    apply_theme(fig, ax)

    # Draw Unit Circle |z| = 1
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=TEAL_ACCENT, linestyle="-", linewidth=2.0, label="Unit Circle |z| = 1")
    
    # Axes
    ax.axhline(0, color=GRID_COLOR, linestyle="-", linewidth=1.2)
    ax.axvline(0, color=GRID_COLOR, linestyle="-", linewidth=1.2)

    # Shaded unit disk
    unit_disk = patches.Circle((0, 0), 1.0, facecolor=TEAL_ACCENT, alpha=0.06)
    ax.add_patch(unit_disk)

    # Plot Zeros ('o')
    if len(zeros) > 0:
        ax.scatter(
            np.real(zeros), np.imag(zeros),
            s=80, facecolors="none", edgecolors=CYAN_ACCENT, linewidths=2.2,
            label=f"Zeros ({len(zeros)})", zorder=5
        )

    # Plot Poles ('x')
    if len(poles) > 0:
        ax.scatter(
            np.real(poles), np.imag(poles),
            s=80, marker="x", color=CORAL_ACCENT, linewidths=2.5,
            label=f"Poles ({len(poles)})", zorder=6
        )

    # Stability Banner
    status_color = EMERALD_ACCENT if is_stable else CORAL_ACCENT
    status_text = "BIBO STABLE (|p| < 1)" if is_stable else "UNSTABLE (|p| >= 1)"
    ax.text(
        0.05, 0.95, f"Status: {status_text}",
        transform=ax.transAxes,
        fontsize=9.5, fontweight="bold",
        color=status_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK, edgecolor=status_color, alpha=0.9)
    )

    ax.set_xlabel(r"Real Part: $\text{Re}(z)$")
    ax.set_ylabel(r"Imaginary Part: $\text{Im}(z)$")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    
    # Set equal aspect ratio and limits
    max_radius = 1.4
    if len(poles) > 0:
        max_radius = max(max_radius, np.max(np.abs(poles)) * 1.25)
    if len(zeros) > 0:
        max_radius = max(max_radius, np.max(np.abs(zeros)) * 1.25)
    max_radius = min(max_radius, 3.0)  # Avoid extreme blowup

    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_MAIN)
    fig.tight_layout()
    return fig


def plot_frequency_response(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
    cutoff_freq: Optional[float] = None,
    cutoff_freq_high: Optional[float] = None
) -> plt.Figure:
    """Plots Magnitude Response (dB) and Phase Response (degrees) across frequency."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True, dpi=100)
    apply_theme(fig, [ax1, ax2])

    # Magnitude Response
    ax1.plot(freqs, mag_db, color=CYAN_ACCENT, linewidth=1.6, label=r"$|H(e^{j\omega})|$ (dB)")
    ax1.axhline(-3, color=AMBER_ACCENT, linestyle=":", linewidth=1.2, label="-3 dB Cutoff Reference")
    if cutoff_freq is not None:
        ax1.axvline(cutoff_freq, color=CORAL_ACCENT, linestyle="--", linewidth=1.4, label=f"fc1 = {cutoff_freq:.0f} Hz")
    if cutoff_freq_high is not None:
        ax1.axvline(cutoff_freq_high, color=CORAL_ACCENT, linestyle="--", linewidth=1.4, label=f"fc2 = {cutoff_freq_high:.0f} Hz")
    
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Filter Frequency Response: Magnitude & Phase", fontsize=11, fontweight="bold")
    ax1.set_ylim([max(-90.0, float(np.min(mag_db) - 5)), 5.0])
    ax1.legend(loc="lower right", facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_MAIN)

    # Phase Response
    ax2.plot(freqs, phase_deg, color=EMERALD_ACCENT, linewidth=1.4, label=r"$\angle H(e^{j\omega})$ (deg)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.legend(loc="lower right", facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_MAIN)

    fig.tight_layout()
    return fig


def plot_dual_spectrogram(
    x_orig: np.ndarray,
    y_enh: np.ndarray,
    fs: int
) -> plt.Figure:
    """Generates side-by-side time-frequency spectrograms showing noise removal."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, dpi=100)
    apply_theme(fig, [ax1, ax2])

    N = min(len(x_orig), len(y_enh))
    nperseg = min(512, max(64, N // 40))

    # STFT for Original
    f1, t1, Sxx1 = spectrogram(x_orig[:N], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx1_db = 10 * np.log10(np.maximum(Sxx1, 1e-10))

    # STFT for Enhanced
    f2, t2, Sxx2 = spectrogram(y_enh[:N], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx2_db = 10 * np.log10(np.maximum(Sxx2, 1e-10))

    vmin = max(np.max(Sxx1_db) - 70.0, -80.0)
    vmax = np.max(Sxx1_db)

    im1 = ax1.pcolormesh(t1, f1, Sxx1_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    ax1.set_title("Input / Noisy Audio Spectrogram (Time-Frequency)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency (Hz)")

    im2 = ax2.pcolormesh(t2, f2, Sxx2_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    ax2.set_title("Enhanced Output Spectrogram (Filtered)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical", pad=0.02, aspect=25)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.set_label("Power Spectral Density (dB/Hz)", color=TEXT_MAIN, fontsize=8.5)

    fig.tight_layout()
    return fig


def plot_z_plane_surface_2d(
    r_vals: np.ndarray,
    theta_vals: np.ndarray,
    mag_surface_db: np.ndarray
) -> plt.Figure:
    """Plots a 2D polar/Cartesian heat map of |X(z)| in the complex z-plane."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    apply_theme(fig, ax)

    R, THETA = np.meshgrid(r_vals, theta_vals, indexing="ij")
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    vmin = np.percentile(mag_surface_db, 5)
    vmax = np.percentile(mag_surface_db, 98)

    mesh = ax.pcolormesh(X, Y, mag_surface_db, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    
    # Highlight unit circle
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=CORAL_ACCENT, linestyle="--", linewidth=2.0, label="Unit Circle |z|=1 (DTFT)")

    ax.axhline(0, color=GRID_COLOR, linestyle="-", linewidth=1.0)
    ax.axvline(0, color=GRID_COLOR, linestyle="-", linewidth=1.0)

    ax.set_xlabel(r"$\text{Re}(z)$")
    ax.set_ylabel(r"$\text{Im}(z)$")
    ax.set_title(r"Z-Transform Magnitude Surface $|X(z)|$ (dB)", fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_MAIN)

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.03, aspect=20)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.set_label(r"$|X(z)|$ (dB)", color=TEXT_MAIN, fontsize=8.5)

    fig.tight_layout()
    return fig
