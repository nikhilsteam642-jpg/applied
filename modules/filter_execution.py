"""
Filter Execution & Digital Audio Filtering Module
Applies the designed digital filter to discrete audio samples x[n].
Supports causal Direct Form II filtering (lfilter) and Zero-Phase Forward-Backward filtering (filtfilt).
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Dict, Any


def apply_digital_filter(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    method: str = "filtfilt",
    preserve_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Filters the discrete audio signal x[n] with digital filter (b, a).
    
    Args:
        x (np.ndarray): Discrete input audio sequence x[n].
        b (np.ndarray): Numerator filter coefficients.
        a (np.ndarray): Denominator filter coefficients.
        method (str): "filtfilt" (Zero-Phase) or "lfilter" (Causal LTI).
        preserve_scale (bool): Whether to safely normalize output amplitude.
        
    Returns:
        y (np.ndarray): Enhanced/filtered audio sequence y[n].
        residual (np.ndarray): Removed noise/component signal e[n] = x[n] - y[n].
        info (dict): Execution details, method description.
    """
    N = len(x)
    if N == 0:
        return np.array([]), np.array([]), {}

    # Ensure float64
    x_input = x.astype(np.float64)

    if method == "filtfilt":
        # Forward-backward zero-phase filtering
        # Check that signal length is sufficient for filtfilt padding
        padlen = 3 * max(len(a), len(b))
        if N > padlen:
            try:
                y = signal.filtfilt(b, a, x_input)
                desc = "Zero-Phase Forward-Backward Filtering (filtfilt): Zero phase distortion, doubled order magnitude."
            except Exception:
                y = signal.lfilter(b, a, x_input)
                desc = "Causal Direct Form II Transposed (lfilter - fallback due to condition number)."
        else:
            y = signal.lfilter(b, a, x_input)
            desc = "Causal Direct Form II Transposed (lfilter - short signal)."
    else:
        # Causal LTI difference equation filtering (Direct Form II Transposed)
        y = signal.lfilter(b, a, x_input)
        desc = "Causal Direct Form II Transposed Filtering (lfilter): Simulates real-time hardware DSP with causal delay."

    # Residual removed signal: e[n] = x[n] - y[n]
    residual = x_input - y

    # Safe Peak Normalization for playback if needed
    if preserve_scale:
        peak_y = np.max(np.abs(y))
        if peak_y > 1.0:
            y = y / (peak_y + 1e-6) * 0.95

    info = {
        "method": method,
        "description": desc,
        "num_samples_processed": N,
        "output_peak": float(np.max(np.abs(y))),
        "output_rms": float(np.sqrt(np.mean(y**2))),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
    }

    return y, residual, info


def compute_impulse_and_step_response(
    b: np.ndarray,
    a: np.ndarray,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Computes discrete unit impulse response h[n] and step response s[n]:
    h[n] = Z^{-1}{H(z)}, \quad s[n] = \sum_{k=0}^n h[k]
    """
    impulse = np.zeros(n_points)
    impulse[0] = 1.0
    h_n = signal.lfilter(b, a, impulse)

    step = np.ones(n_points)
    s_n = signal.lfilter(b, a, step)

    n_axis = np.arange(n_points)

    return {
        "n_axis": n_axis,
        "impulse_response": h_n,
        "step_response": s_n,
    }
