"""
Discrete Fourier Transform & Spectral Analysis Module
Implements discrete-time frequency analysis, single-sided FFT, magnitude/phase spectra,
peak detection, spectral centroid, spectral bandwidth, and spectral flatness.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Dict, Any, List


def compute_fft(
    signal: np.ndarray,
    fs: int,
    n_fft: int = None,
    window: str = "hann"
) -> Dict[str, Any]:
    """
    Computes single-sided Discrete Fourier Transform (DFT) via FFT:
    X[k] = \sum_{n=0}^{N-1} x[n] \cdot w[n] \cdot e^{-j 2\pi k n / N}
    
    Returns a dictionary containing:
        freqs (np.ndarray): Frequency axis in Hz [0, fs/2]
        mag_linear (np.ndarray): Properly scaled single-sided magnitude
        mag_db (np.ndarray): Magnitude in decibels (dB)
        phase_rad (np.ndarray): Phase in radians [-pi, pi]
        psd (np.ndarray): Power Spectral Density
        dominant_peaks (list): Top prominent frequency peaks
    """
    N = len(signal)
    if N == 0:
        return {"freqs": np.array([]), "mag_linear": np.array([]), "mag_db": np.array([])}

    if n_fft is None:
        # Next power of 2 for FFT efficiency, at least N
        n_fft = max(2048, int(2 ** np.ceil(np.log2(N))))

    # Apply windowing to reduce spectral leakage
    if window == "hann":
        win = np.hanning(N)
        win_scale = np.sum(win)
    elif window == "hamming":
        win = np.hamming(N)
        win_scale = np.sum(win)
    elif window == "blackman":
        win = np.blackman(N)
        win_scale = np.sum(win)
    else:
        win = np.ones(N)
        win_scale = N

    windowed_signal = signal * win
    
    # Compute real FFT (efficient for real-valued audio signals)
    fft_complex = np.fft.rfft(windowed_signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Magnitude scaling: 2 / win_scale for single-sided spectrum (1 / win_scale for DC and Nyquist)
    mag_linear = np.abs(fft_complex) * (2.0 / win_scale)
    mag_linear[0] /= 2.0  # DC component
    if len(mag_linear) % 2 == 0:
        mag_linear[-1] /= 2.0  # Nyquist component

    # Logarithmic magnitude (dB)
    eps = 1e-12
    mag_db = 20.0 * np.log10(np.maximum(mag_linear, eps))
    
    # Normalize peak dB to 0 dB for clear relative spectral comparison
    peak_db = np.max(mag_db)
    mag_db_normalized = mag_db - peak_db

    # Phase spectrum
    phase_rad = np.angle(fft_complex)

    # Power Spectral Density (PSD)
    psd = (np.abs(fft_complex) ** 2) / (fs * win_scale)

    # Detect dominant peaks
    peaks_info = find_dominant_peaks(freqs, mag_db_normalized, top_k=6)
    
    # Compute spectral moments
    spectral_stats = compute_spectral_moments(freqs, mag_linear)

    return {
        "freqs": freqs,
        "mag_linear": mag_linear,
        "mag_db": mag_db,
        "mag_db_normalized": mag_db_normalized,
        "phase_rad": phase_rad,
        "psd": psd,
        "fft_complex": fft_complex,
        "n_fft": n_fft,
        "dominant_peaks": peaks_info,
        "spectral_stats": spectral_stats,
    }


def find_dominant_peaks(
    freqs: np.ndarray,
    mag_db_norm: np.ndarray,
    top_k: int = 6,
    min_distance_hz: float = 40.0
) -> List[Dict[str, float]]:
    """
    Finds prominent spectral peaks above threshold in the normalized dB spectrum.
    """
    if len(freqs) < 2:
        return []
    
    df = freqs[1] - freqs[0]
    min_dist_bins = max(1, int(min_distance_hz / df))

    # Look for peaks with at least 3 dB prominence above background
    peak_indices, properties = find_peaks(
        mag_db_norm,
        height=-60.0,
        distance=min_dist_bins,
        prominence=2.5
    )

    if len(peak_indices) == 0:
        # Fallback to top max bins if find_peaks returned none
        peak_indices = np.argsort(mag_db_norm)[-top_k:]

    # Sort peaks by peak amplitude descending
    sorted_idx = peak_indices[np.argsort(mag_db_norm[peak_indices])[::-1]][:top_k]

    peaks = []
    for idx in sorted_idx:
        peaks.append({
            "frequency_hz": float(freqs[idx]),
            "magnitude_db_rel": float(mag_db_norm[idx]),
            "index": int(idx),
        })

    # Sort in ascending frequency order for nice display
    peaks.sort(key=lambda x: x["frequency_hz"])
    return peaks


def compute_spectral_moments(freqs: np.ndarray, mag_linear: np.ndarray) -> Dict[str, float]:
    """
    Calculates statistical spectral features:
    - Spectral Centroid: \mu_f = \sum (f \cdot |X(f)|) / \sum |X(f)|
    - Spectral Bandwidth: \sigma_f = \sqrt{ \sum ((f - \mu_f)^2 \cdot |X(f)|) / \sum |X(f)| }
    - Spectral Flatness: Geometric Mean / Arithmetic Mean of Power Spectrum
    - Spectral Rolloff (85% energy frequency)
    """
    total_mag = np.sum(mag_linear)
    if total_mag < 1e-12:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff_85_hz": 0.0,
        }

    # Centroid (Center of mass of spectrum)
    centroid = float(np.sum(freqs * mag_linear) / total_mag)

    # Bandwidth / Spread
    variance = np.sum(((freqs - centroid) ** 2) * mag_linear) / total_mag
    bandwidth = float(np.sqrt(max(0.0, variance)))

    # Spectral Flatness (Wiener entropy)
    power = mag_linear ** 2 + 1e-15
    geom_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    flatness = float(geom_mean / arith_mean) if arith_mean > 0 else 0.0

    # 85% Rolloff frequency
    cum_energy = np.cumsum(power)
    threshold = 0.85 * cum_energy[-1]
    rolloff_idx = np.searchsorted(cum_energy, threshold)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    rolloff_hz = float(freqs[rolloff_idx])

    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_flatness": flatness,
        "spectral_rolloff_85_hz": rolloff_hz,
    }
