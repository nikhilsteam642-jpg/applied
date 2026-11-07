"""
Z-Transform Mathematical Modeling Module
Implements discrete-time Z-transform analysis for finite-length audio sequences:
X(z) = \sum_{n=0}^{N-1} x[n] z^{-n} = \frac{x[0]z^{N-1} + x[1]z^{N-2} + ... + x[N-1]}{z^{N-1}}
Includes polynomial zero extraction, 2D/3D complex plane manifold evaluation,
and unit circle DTFT intersection.
"""

import numpy as np
from typing import Tuple, Dict, Any, List


def extract_analysis_window(
    signal: np.ndarray,
    window_length: int = 64,
    start_ratio: float = 0.3
) -> np.ndarray:
    """
    Extracts a representative finite discrete window x[n] (length N) from the audio signal.
    """
    N = len(signal)
    if N <= window_length:
        return signal.copy()
    
    # Select a region with meaningful acoustic energy (avoid silence)
    start_idx = int(start_ratio * (N - window_length))
    window = signal[start_idx : start_idx + window_length]
    
    # If energy is too low, find the peak energy window
    if np.max(np.abs(window)) < 0.05:
        peak_idx = int(np.argmax(np.abs(signal)))
        start_idx = max(0, min(peak_idx - window_length // 2, N - window_length))
        window = signal[start_idx : start_idx + window_length]
        
    return window


def compute_signal_zeros(x_window: np.ndarray) -> Dict[str, Any]:
    """
    Computes the roots (zeros) of the finite-duration discrete audio polynomial:
    P(z) = x[0] z^{N-1} + x[1] z^{N-2} + ... + x[N-1] = 0
    
    The poles of a finite-duration causal signal are trivially N-1 poles at z = 0.
    The zeros represent the complex frequencies where the signal's spectral content vanishes.
    """
    N = len(x_window)
    if N < 2:
        return {"zeros": np.array([]), "poles": np.array([0.0])}
    
    # Polynomial coefficients in descending powers of z:
    # x[0] z^(N-1) + x[1] z^(N-2) + ... + x[N-1]
    poly_coeffs = x_window.copy()
    
    # Avoid leading zeros which cause root finding issues
    first_nonzero = np.argmax(np.abs(poly_coeffs) > 1e-6)
    poly_coeffs = poly_coeffs[first_nonzero:]
    
    if len(poly_coeffs) < 2:
        zeros = np.array([], dtype=complex)
    else:
        try:
            zeros = np.roots(poly_coeffs)
        except Exception:
            zeros = np.array([], dtype=complex)
            
    # Poles for finite causal sequence are at origin z = 0
    num_poles = max(1, len(poly_coeffs) - 1)
    poles = np.zeros(num_poles, dtype=complex)
    
    # Compute radii and angles for zeros
    zero_radii = np.abs(zeros)
    zero_angles_rad = np.angle(zeros)
    
    return {
        "zeros": zeros,
        "poles": poles,
        "zero_radii": zero_radii,
        "zero_angles_rad": zero_angles_rad,
        "num_zeros": len(zeros),
        "num_poles": len(poles),
        "polynomial_degree": len(poly_coeffs) - 1,
    }


def evaluate_z_plane_surface(
    x_window: np.ndarray,
    r_min: float = 0.5,
    r_max: float = 1.5,
    n_r: int = 40,
    n_theta: int = 150
) -> Dict[str, Any]:
    """
    Evaluates the Laurent polynomial |X(z)| = |\sum x[n] z^{-n}| across a polar grid in the complex z-plane:
    z = r \cdot e^{j \theta}, \quad r \in [r_min, r_max], \quad \theta \in [-\pi, \pi]
    
    Shows how the slice at r = 1 (|z| = 1) yields the Discrete-Time Fourier Transform (DTFT).
    """
    N = len(x_window)
    r_vals = np.linspace(r_min, r_max, n_r)
    theta_vals = np.linspace(-np.pi, np.pi, n_theta)
    
    R, THETA = np.meshgrid(r_vals, theta_vals, indexing="ij")
    Z = R * np.exp(1j * THETA)
    
    # Vectorized evaluation of polynomial: X(z) = \sum_{n=0}^{N-1} x[n] * z^{-n}
    # Shape of Z is (n_r, n_theta)
    X_z = np.zeros_like(Z, dtype=complex)
    for n in range(N):
        if np.abs(x_window[n]) > 1e-12:
            X_z += x_window[n] * (Z ** (-n))
            
    mag_surface = np.abs(X_z)
    
    # Log scale for clear visualization
    eps = 1e-6
    mag_surface_db = 20.0 * np.log10(np.maximum(mag_surface, eps))
    
    # Extract exact unit circle slice (r closest to 1.0)
    idx_unit = np.argmin(np.abs(r_vals - 1.0))
    unit_circle_slice = mag_surface[idx_unit, :]
    unit_circle_slice_db = mag_surface_db[idx_unit, :]
    
    # Cartesian coordinates for 3D plotting
    X_real = R * np.cos(THETA)
    Y_imag = R * np.sin(THETA)
    
    return {
        "r_vals": r_vals,
        "theta_vals": theta_vals,
        "X_real": X_real,
        "Y_imag": Y_imag,
        "mag_surface": mag_surface,
        "mag_surface_db": mag_surface_db,
        "unit_circle_slice": unit_circle_slice,
        "unit_circle_slice_db": unit_circle_slice_db,
        "r_unit_actual": float(r_vals[idx_unit]),
    }


def format_polynomial_latex(x_window: np.ndarray, max_terms: int = 6) -> str:
    """
    Formats the finite discrete sequence Laurent series into LaTeX representation.
    """
    N = len(x_window)
    if N == 0:
        return "X(z) = 0"
    
    terms = []
    # Show first few non-zero terms
    limit = min(N, max_terms)
    for n in range(limit):
        val = x_window[n]
        if n == 0:
            terms.append(f"{val:+.3f}")
        elif n == 1:
            terms.append(f"{val:+.3f} z^{{-1}}")
        else:
            terms.append(f"{val:+.3f} z^{{-{n}}}")
            
    if N > max_terms:
        # Show ellipsis and last term
        terms.append(r"\dots")
        last_val = x_window[-1]
        terms.append(f"{last_val:+.3f} z^{{-{N-1}}}")
        
    formula = "X(z) = " + " ".join(terms)
    # Clean leading '+' if first term is positive
    formula = formula.replace("X(z) = +", "X(z) = ")
    return formula
