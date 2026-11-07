"""
Verification Test Script for Audio Signal Enhancement Pipeline
Tests all mathematical modules, DSP functions, and visualization routines.
"""

import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.audio_processing import (
    generate_synthetic_signal,
    inject_noise,
    audio_to_wav_bytes,
    normalize_signal
)
from modules.fft_analysis import compute_fft, compute_spectral_moments
from modules.z_transform import (
    extract_analysis_window,
    compute_signal_zeros,
    evaluate_z_plane_surface,
    format_polynomial_latex
)
from modules.filter_design import design_digital_filter
from modules.filter_execution import apply_digital_filter, compute_impulse_and_step_response
from modules.performance import evaluate_audio_performance
from modules.visualization import (
    plot_waveform_single,
    plot_waveform_comparison,
    plot_fft_spectrum,
    plot_fft_comparison,
    plot_pole_zero_diagram,
    plot_frequency_response,
    plot_dual_spectrogram,
    plot_z_plane_surface_2d
)

def run_all_tests():
    print("==================================================================")
    print("RUNNING DSP & MATHEMATICS CAPSTONE TEST SUITE")
    print("==================================================================")

    # 1. Test Audio Synthesis & Noise Injection
    print("\n[TEST 1] Testing Audio Synthesis and Noise Injection...")
    fs = 16000
    t, x_clean, meta = generate_synthetic_signal("Vocal / Speech Formant", duration_sec=2.0, fs=fs)
    assert len(x_clean) == 32000, f"Expected 32000 samples, got {len(x_clean)}"
    assert np.max(np.abs(x_clean)) <= 1.0, "Audio is not properly normalized"

    x_noisy, noise, n_info = inject_noise(x_clean, noise_type="50 Hz Powerline Hum", snr_db=10.0, fs=fs)
    assert len(x_noisy) == len(x_clean), "Noisy audio length mismatch"
    assert abs(n_info["actual_snr_db"] - 10.0) < 0.2, f"Target SNR mismatch: {n_info['actual_snr_db']}"
    print(f"  ✓ Clean Signal: {meta['description']}")
    print(f"  ✓ Injected Noise: {n_info['description']} (Actual SNR: {n_info['actual_snr_db']:.2f} dB)")

    # 2. Test WAV Audio Encoding
    print("\n[TEST 2] Testing In-Memory WAV Bytes Generation...")
    wav_bytes = audio_to_wav_bytes(x_noisy, fs)
    assert len(wav_bytes) > 1000, "WAV bytes generation failed"
    print(f"  ✓ Encoded WAV buffer size: {len(wav_bytes)} bytes")

    # 3. Test Discrete Fourier Transform (FFT)
    print("\n[TEST 3] Testing Discrete Fourier Transform (FFT) Analysis...")
    fft_res = compute_fft(x_noisy, fs=fs)
    assert len(fft_res["freqs"]) > 1000, "FFT frequencies missing"
    assert len(fft_res["dominant_peaks"]) > 0, "No peaks detected in FFT"
    print(f"  ✓ FFT Bins: {fft_res['n_fft']}, Detected Peaks: {len(fft_res['dominant_peaks'])}")
    for pk in fft_res["dominant_peaks"][:3]:
        print(f"    - Peak at {pk['frequency_hz']:.1f} Hz (Mag: {pk['magnitude_db_rel']:.1f} dB)")

    # 4. Test Z-Transform Modeling
    print("\n[TEST 4] Testing Z-Transform Laurent Polynomial & Complex Zero Extraction...")
    x_win = extract_analysis_window(x_noisy, window_length=32, start_ratio=0.3)
    z_res = compute_signal_zeros(x_win)
    assert z_res["num_zeros"] > 0, "Failed to extract polynomial zeros"
    latex_poly = format_polynomial_latex(x_win, max_terms=5)
    assert "X(z)" in latex_poly, "LaTeX polynomial formatting error"
    surface = evaluate_z_plane_surface(x_win, r_min=0.5, r_max=1.5, n_r=20, n_theta=50)
    assert surface["mag_surface"].shape == (20, 50), "Z-plane surface evaluation shape mismatch"
    print(f"  ✓ Extracted {z_res['num_zeros']} zeros in complex Z-plane")
    print(f"  ✓ Laurent Polynomial: {latex_poly[:45]}...")

    # 5. Test Filter Design & Stability Verification
    print("\n[TEST 5] Testing Filter Design in Z-Domain (Butterworth, Chebyshev, Notch, FIR)...")
    filter_configs = [
        ("Butterworth (IIR)", "High-pass", 80.0, 4),
        ("Chebyshev Type I (IIR)", "Low-pass", 1500.0, 4),
        ("Digital Notch Filter (IIR)", "Notch (Interference Removal)", 50.0, 2),
        ("FIR Windowed (Hamming)", "Low-pass", 2000.0, 15),
    ]
    for fam, ftype, fc, order in filter_configs:
        f_design = design_digital_filter(filter_family=fam, filter_type=ftype, cutoff_freq=fc, filter_order=order, fs=fs)
        assert f_design["is_stable"], f"Filter {fam} was unexpectedly flagged unstable"
        assert len(f_design["b"]) > 0 and len(f_design["a"]) > 0, "Missing filter coefficients"
        print(f"  ✓ {fam} ({ftype}, fc={fc}Hz): BIBO Stable! (Max Pole |p| = {f_design['max_pole_mag']:.4f})")

    # 6. Test Filter Application
    print("\n[TEST 6] Testing Digital Filtering Application (filtfilt & lfilter)...")
    highpass_design = design_digital_filter(filter_family="Butterworth (IIR)", filter_type="High-pass", cutoff_freq=80.0, filter_order=4, fs=fs)
    y_enh, residual, exec_info = apply_digital_filter(x_noisy, highpass_design["b"], highpass_design["a"], method="filtfilt")
    assert len(y_enh) == len(x_noisy), "Filtered output length mismatch"
    print(f"  ✓ Filtered {exec_info['num_samples_processed']} samples via {exec_info['method']}")
    print(f"  ✓ Output RMS: {exec_info['output_rms']:.4f}, Residual Noise RMS: {exec_info['residual_rms']:.4f}")

    # 7. Test Performance Evaluation
    print("\n[TEST 7] Testing Performance Evaluation & SNR Improvement...")
    perf = evaluate_audio_performance(
        x_orig=x_noisy,
        y_enh=y_enh,
        x_clean=x_clean,
        fs=fs,
        filter_type="High-pass",
        cutoff_freq=80.0
    )
    assert perf["has_ground_truth"] is True, "Ground truth flag should be True"
    assert perf["delta_snr_db"] is not None, "Delta SNR should be calculated"
    print(f"  ✓ Input SNR: {perf['snr_in_db']:.2f} dB")
    print(f"  ✓ Enhanced SNR: {perf['snr_out_db']:.2f} dB")
    print(f"  ✓ Δ SNR Improvement: {perf['delta_snr_db']:+.2f} dB")
    print(f"  ✓ Stopband Attenuation: {perf['stopband_attenuation_db']:.2f} dB")
    print(f"  ✓ MSE Reduction: {perf['mse_reduction_pct']:.2f}%")

    # 8. Test Visualizations Rendering
    print("\n[TEST 8] Testing Scientific Visualization Plot Generation...")
    fig1 = plot_waveform_comparison(t, x_noisy, y_enh)
    fig2 = plot_fft_comparison(fft_res["freqs"], fft_res["mag_db_normalized"], fft_res["mag_db_normalized"])
    fig3 = plot_pole_zero_diagram(highpass_design["zeros"], highpass_design["poles"], highpass_design["is_stable"])
    fig4 = plot_frequency_response(highpass_design["freqs"], highpass_design["mag_db"], highpass_design["phase_deg"])
    fig5 = plot_dual_spectrogram(x_noisy, y_enh, fs)
    fig6 = plot_z_plane_surface_2d(surface["r_vals"], surface["theta_vals"], surface["mag_surface_db"])
    print("  ✓ All 6 publication-grade figures generated successfully without errors")

    print("\n==================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY (100% OPERATIONAL)")
    print("==================================================================")

if __name__ == "__main__":
    run_all_tests()
