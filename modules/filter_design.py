"""
Digital Filter Design & Z-Domain Transfer Function Module
Implements FIR (Windowed) and IIR (Butterworth, Chebyshev I & II, Elliptic, Notch) filter synthesis.
Computes Z-domain transfer functions H(z) = B(z)/A(z), pole-zero locations, BIBO stability metrics,
and frequency/phase responses.
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Dict, Any, List, Optional


def design_digital_filter(
    filter_family: str = "Butterworth (IIR)",
    filter_type: str = "Low-pass",
    cutoff_freq: float = 1000.0,
    cutoff_freq_high: Optional[float] = 3000.0,
    filter_order: int = 4,
    fs: int = 16000,
    rp: float = 1.0,      # Passband ripple (dB) for Chebyshev I / Elliptic
    rs: float = 40.0,     # Stopband attenuation (dB) for Chebyshev II / Elliptic
    notch_q: float = 30.0 # Quality factor for Notch filter
) -> Dict[str, Any]:
    """
    Synthesizes digital filter coefficients b and a in the Z-domain.
    Validates Nyquist limits and computes poles, zeros, and stability.
    """
    nyquist = fs / 2.0
    
    # Defensive Nyquist validation
    if filter_type in ["Low-pass", "High-pass", "Notch (Interference Removal)"]:
        cutoff_freq = max(10.0, min(cutoff_freq, nyquist - 20.0))
        Wn = cutoff_freq / nyquist
    else:  # Band-pass or Band-stop
        cutoff_freq = max(10.0, min(cutoff_freq, nyquist - 60.0))
        cutoff_freq_high = max(cutoff_freq + 20.0, min(cutoff_freq_high, nyquist - 20.0))
        Wn = [cutoff_freq / nyquist, cutoff_freq_high / nyquist]

    # Map friendly filter type names to scipy conventions
    btype_map = {
        "Low-pass": "lowpass",
        "High-pass": "highpass",
        "Band-pass": "bandpass",
        "Band-stop": "bandstop",
        "Notch (Interference Removal)": "notch",
    }
    btype = btype_map.get(filter_type, "lowpass")

    try:
        if filter_family == "Butterworth (IIR)":
            b, a = signal.butter(filter_order, Wn, btype=btype, output="ba")
            family_desc = "Butterworth IIR Filter (Maximally Flat Passband)"

        elif filter_family == "Chebyshev Type I (IIR)":
            b, a = signal.cheby1(filter_order, rp, Wn, btype=btype, output="ba")
            family_desc = f"Chebyshev Type I IIR (Passband Ripple: {rp:.1f} dB, Sharp Roll-off)"

        elif filter_family == "Chebyshev Type II (IIR)":
            b, a = signal.cheby2(filter_order, rs, Wn, btype=btype, output="ba")
            family_desc = f"Chebyshev Type II IIR (Monotonic Passband, Stopband Rejection: {rs:.1f} dB)"

        elif filter_family == "Elliptic (Cauer) (IIR)":
            b, a = signal.ellip(filter_order, rp, rs, Wn, btype=btype, output="ba")
            family_desc = f"Elliptic IIR (Steepest Transition Band, {rp:.1f} dB ripple, {rs:.1f} dB stopband)"

        elif filter_family.startswith("FIR Windowed"):
            # Determine window type
            if "Hamming" in filter_family:
                win_type = "hamming"
            elif "Hann" in filter_family:
                win_type = "hann"
            elif "Blackman" in filter_family:
                win_type = "blackman"
            else:
                win_type = "hamming"

            # FIR filter order must yield odd number of taps for highpass/bandstop
            numtaps = filter_order * 4 + 1  # FIR needs higher order for comparable sharpness
            if btype in ["highpass", "bandstop"] and numtaps % 2 == 0:
                numtaps += 1

            # firwin cutoffs normalized to [0, nyquist] (pass_zero determines type)
            pass_zero_map = {
                "lowpass": True,
                "highpass": False,
                "bandpass": False,
                "bandstop": True,
            }
            pass_zero = pass_zero_map.get(btype, True)
            
            b = signal.firwin(numtaps, Wn, window=win_type, pass_zero=pass_zero)
            a = np.array([1.0], dtype=np.float64)
            family_desc = f"FIR Windowed Filter ({win_type.capitalize()} Window, {numtaps} Taps, Linear Phase)"

        elif filter_family == "Digital Notch Filter (IIR)":
            # Direct notch filter synthesis: places conjugate zeros on unit circle at e^(+/- j \omega_0)
            b, a = signal.iirnotch(cutoff_freq, notch_q, fs=fs)
            family_desc = f"IIR Digital Notch Filter (Center: {cutoff_freq:.1f} Hz, Q-Factor: {notch_q:.1f})"

        else:
            # Fallback Butterworth
            b, a = signal.butter(4, 0.2, btype="lowpass", output="ba")
            family_desc = "Butterworth Lowpass (Default)"

    except Exception as e:
        # Fallback safe filter
        b, a = signal.butter(2, 0.2, btype="lowpass", output="ba")
        family_desc = f"Fallback Safe Butterworth (Error: {str(e)})"

    # Extract Z-plane Poles and Zeros: H(z) = B(z) / A(z)
    zeros, poles, k_gain = signal.tf2zpk(b, a)
    
    # Check BIBO Stability: |p_k| < 1 for all poles
    pole_magnitudes = np.abs(poles) if len(poles) > 0 else np.array([0.0])
    max_pole_mag = float(np.max(pole_magnitudes)) if len(pole_magnitudes) > 0 else 0.0
    is_stable = bool(max_pole_mag < 1.000)
    stability_margin = float(1.0 - max_pole_mag)

    # Compute Complex Frequency Response H(e^(j\omega))
    w, h_complex = signal.freqz(b, a, worN=1024, fs=fs)
    freqs = w
    mag_linear = np.abs(h_complex)
    mag_db = 20.0 * np.log10(np.maximum(mag_linear, 1e-10))
    phase_rad = np.unwrap(np.angle(h_complex))
    phase_deg = np.degrees(phase_rad)

    # Calculate Group Delay: \tau_g(\omega) = -d\phi(\omega)/d\omega
    try:
        w_gd, gd = signal.group_delay((b, a), w=1024, fs=fs)
    except Exception:
        w_gd = freqs
        gd = np.zeros_like(freqs)

    # Generate LaTeX formulas
    hz_latex = format_transfer_function_latex(b, a)
    diff_eq_latex = format_difference_equation_latex(b, a)

    return {
        "b": b,
        "a": a,
        "zeros": zeros,
        "poles": poles,
        "gain": float(k_gain),
        "is_stable": is_stable,
        "max_pole_mag": max_pole_mag,
        "stability_margin": stability_margin,
        "freqs": freqs,
        "mag_linear": mag_linear,
        "mag_db": mag_db,
        "phase_rad": phase_rad,
        "phase_deg": phase_deg,
        "group_delay_samples": gd,
        "group_delay_freqs": w_gd,
        "filter_family": filter_family,
        "filter_type": filter_type,
        "cutoff_freq": cutoff_freq,
        "cutoff_freq_high": cutoff_freq_high,
        "filter_order": filter_order,
        "family_desc": family_desc,
        "hz_latex": hz_latex,
        "diff_eq_latex": diff_eq_latex,
        "num_zeros": len(zeros),
        "num_poles": len(poles),
    }


def format_transfer_function_latex(b: np.ndarray, a: np.ndarray, max_terms: int = 5) -> str:
    """
    Formats the rational transfer function H(z) = B(z)/A(z) into LaTeX.
    """
    def format_poly(coeffs, var="z"):
        terms = []
        for i, c in enumerate(coeffs[:max_terms]):
            if np.abs(c) < 1e-6:
                continue
            if i == 0:
                terms.append(f"{c:.3f}")
            elif i == 1:
                terms.append(f"{c:+.3f}{var}^{{-1}}")
            else:
                terms.append(f"{c:+.3f}{var}^{{-{i}}}")
        if len(coeffs) > max_terms:
            terms.append(r"\dots")
        if not terms:
            return "0"
        poly_str = " ".join(terms)
        return poly_str.replace("+-", "-").replace("+ -", "- ")

    num_str = format_poly(b)
    den_str = format_poly(a)
    return f"H(z) = \\frac{{{num_str}}}{{{den_str}}}"


def format_difference_equation_latex(b: np.ndarray, a: np.ndarray, max_terms: int = 4) -> str:
    """
    Formats the discrete LCCDE difference equation:
    y[n] = \sum b_k x[n-k] - \sum a_k y[n-k]
    """
    x_terms = []
    for k, bk in enumerate(b[:max_terms]):
        if np.abs(bk) < 1e-6:
            continue
        if k == 0:
            x_terms.append(f"{bk:.3f}x[n]")
        else:
            x_terms.append(f"{bk:+.3f}x[n-{k}]")
    if len(b) > max_terms:
        x_terms.append(r"\dots")

    y_terms = []
    # a[0] is always 1.0 (normalized), so feedback starts from a[1]
    for k, ak in enumerate(a[1:max_terms], start=1):
        if np.abs(ak) < 1e-6:
            continue
        # Subtract a_k y[n-k]
        y_terms.append(f"{-ak:+.3f}y[n-{k}]")
    if len(a) > max_terms:
        y_terms.append(r"\dots")

    all_terms = x_terms + y_terms
    if not all_terms:
        return "y[n] = 0"
    eq_str = " ".join(all_terms).replace("+-", "- ").replace("+ -", "- ")
    return f"y[n] = {eq_str}"
