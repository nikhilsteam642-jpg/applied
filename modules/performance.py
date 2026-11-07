"""
Performance Evaluation & Audio Quality Metrics Module
Calculates mathematical metrics: RMS, Peak, Crest Factor, Energy, MSE, True SNR,
Delta SNR improvement, and stopband proxy attenuation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List


def evaluate_audio_performance(
    x_orig: np.ndarray,
    y_enh: np.ndarray,
    x_clean: Optional[np.ndarray] = None,
    fs: int = 16000,
    filter_type: str = "Low-pass",
    cutoff_freq: float = 1000.0,
    cutoff_freq_high: Optional[float] = 3000.0,
) -> Dict[str, Any]:
    """
    Computes rigorous audio quality and mathematical enhancement metrics.
    Handles both known-reference cases (synthetic + injected noise) and arbitrary upload cases.
    """
    N = min(len(x_orig), len(y_enh))
    x_o = x_orig[:N]
    y_e = y_enh[:N]

    # 1. Fundamental Time-Domain Statistics
    rms_orig = float(np.sqrt(np.mean(x_o ** 2)))
    rms_enh = float(np.sqrt(np.mean(y_e ** 2)))
    
    peak_orig = float(np.max(np.abs(x_o))) if N > 0 else 0.0
    peak_enh = float(np.max(np.abs(y_e))) if N > 0 else 0.0
    
    crest_orig = float(peak_orig / (rms_orig + 1e-9))
    crest_enh = float(peak_enh / (rms_enh + 1e-9))
    
    energy_orig = float(np.sum(x_o ** 2))
    energy_enh = float(np.sum(y_e ** 2))
    energy_ratio_db = float(10.0 * np.log10(max(1e-12, energy_enh) / max(1e-12, energy_orig)))

    # Residual Signal: e[n] = x_orig[n] - y_enh[n] (The removed noise/components)
    residual = x_o - y_e
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    residual_energy = float(np.sum(residual ** 2))

    # 2. Spectral Stopband Attenuation (Proxy Metric)
    stopband_attenuation_db, stopband_desc = calculate_spectral_attenuation(
        x_o, y_e, fs, filter_type, cutoff_freq, cutoff_freq_high
    )

    # 3. Ground-Truth SNR Analysis (when clean reference signal exists)
    has_ground_truth = (x_clean is not None and len(x_clean) >= N)
    if has_ground_truth:
        x_c = x_clean[:N]
        p_clean = np.mean(x_c ** 2)

        # Noise in original signal: v_in[n] = x_orig[n] - x_clean[n]
        noise_in = x_o - x_c
        p_noise_in = np.mean(noise_in ** 2)

        # Error in enhanced signal: v_out[n] = y_enh[n] - x_clean[n]
        noise_out = y_e - x_c
        p_noise_out = np.mean(noise_out ** 2)

        snr_in_db = 10.0 * np.log10(p_clean / max(1e-12, p_noise_in))
        snr_out_db = 10.0 * np.log10(p_clean / max(1e-12, p_noise_out))
        delta_snr_db = snr_out_db - snr_in_db

        mse_in = float(p_noise_in)
        mse_out = float(p_noise_out)
        mse_reduction_pct = float(max(0.0, (mse_in - mse_out) / max(1e-12, mse_in) * 100.0))
        
        snr_mode = "Ground-Truth Exact Calculation"
    else:
        # Reference-free proxy
        snr_in_db = None
        snr_out_db = None
        delta_snr_db = None
        mse_in = None
        mse_out = float(np.mean(residual ** 2))
        mse_reduction_pct = None
        snr_mode = "Proxy Metrics (No Ground Truth Reference)"

    return {
        "num_samples": N,
        "rms_orig": rms_orig,
        "rms_enh": rms_enh,
        "peak_orig": peak_orig,
        "peak_enh": peak_enh,
        "crest_orig": crest_orig,
        "crest_enh": crest_enh,
        "energy_orig": energy_orig,
        "energy_enh": energy_enh,
        "energy_ratio_db": energy_ratio_db,
        "residual_rms": residual_rms,
        "residual_energy": residual_energy,
        "stopband_attenuation_db": stopband_attenuation_db,
        "stopband_desc": stopband_desc,
        "has_ground_truth": has_ground_truth,
        "snr_mode": snr_mode,
        "snr_in_db": snr_in_db,
        "snr_out_db": snr_out_db,
        "delta_snr_db": delta_snr_db,
        "mse_in": mse_in,
        "mse_out": mse_out,
        "mse_reduction_pct": mse_reduction_pct,
    }


def calculate_spectral_attenuation(
    x_orig: np.ndarray,
    y_enh: np.ndarray,
    fs: int,
    filter_type: str,
    cutoff_freq: float,
    cutoff_freq_high: Optional[float] = None
) -> Tuple[float, str]:
    """
    Measures the decibel power reduction inside the filter's stopband.
    """
    N = len(x_orig)
    n_fft = max(2048, int(2 ** np.ceil(np.log2(N))))
    
    X_orig = np.abs(np.fft.rfft(x_orig, n=n_fft)) ** 2
    Y_enh = np.abs(np.fft.rfft(y_enh, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Determine stopband mask
    if filter_type == "Low-pass":
        # Stopband is f > cutoff
        mask = freqs > cutoff_freq
        desc = f"High-frequency stopband (f > {cutoff_freq:.0f} Hz)"
    elif filter_type == "High-pass":
        # Stopband is f < cutoff
        mask = freqs < cutoff_freq
        desc = f"Low-frequency stopband (f < {cutoff_freq:.0f} Hz)"
    elif filter_type == "Band-pass":
        f_high = cutoff_freq_high if cutoff_freq_high is not None else cutoff_freq * 2
        mask = (freqs < cutoff_freq) | (freqs > f_high)
        desc = f"Out-of-band rejection (f < {cutoff_freq:.0f} Hz or f > {f_high:.0f} Hz)"
    elif filter_type == "Band-stop":
        f_high = cutoff_freq_high if cutoff_freq_high is not None else cutoff_freq * 2
        mask = (freqs >= cutoff_freq) & (freqs <= f_high)
        desc = f"Band-stop rejection ({cutoff_freq:.0f} Hz - {f_high:.0f} Hz)"
    elif filter_type == "Notch (Interference Removal)":
        # Notch stopband is narrow neighborhood [cutoff - 25 Hz, cutoff + 25 Hz]
        mask = (freqs >= max(0, cutoff_freq - 30.0)) & (freqs <= cutoff_freq + 30.0)
        desc = f"Notch notch rejection ({cutoff_freq - 30:.0f} Hz - {cutoff_freq + 30:.0f} Hz)"
    else:
        mask = np.ones_like(freqs, dtype=bool)
        desc = "Full spectrum"

    e_orig_stop = np.sum(X_orig[mask])
    e_enh_stop = np.sum(Y_enh[mask])

    if e_enh_stop > 1e-12 and e_orig_stop > 1e-12:
        attenuation_db = float(10.0 * np.log10(e_orig_stop / e_enh_stop))
    else:
        attenuation_db = 0.0

    return attenuation_db, desc
