"""
Audio Processing & Synthesis Module
Handles audio loading, format conversion, normalization, synthetic signal generation,
calibrated noise injection, and WAV encoding.
"""

import io
import numpy as np
from scipy.io import wavfile
from typing import Tuple, Dict, Any, Optional

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False


def load_audio_from_file(file_obj) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Load an audio file from an uploaded file object or file path.
    Converts stereo to mono and normalizes safely to [-1.0, 1.0].
    
    Returns:
        signal (np.ndarray): 1D float64 array of discrete samples x[n].
        fs (int): Sampling frequency in Hz.
        info (dict): Metadata including channels, original length, etc.
    """
    data = None
    fs = 16000

    if HAS_SOUNDFILE and sf is not None:
        try:
            data, fs = sf.read(file_obj, dtype="float32")
        except Exception:
            data = None

    if data is None:
        # Fallback to scipy wavfile
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        fs, data = wavfile.read(file_obj)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0

    original_shape = data.shape
    num_channels = 1 if data.ndim == 1 else data.shape[1]

    # Convert to mono if multi-channel
    if num_channels > 1:
        signal = np.mean(data, axis=1)
    else:
        signal = data.copy()

    # Ensure float64 for high-precision DSP calculations
    signal = signal.astype(np.float64)

    # Safe Peak Normalization
    signal = normalize_signal(signal)

    duration = len(signal) / float(fs)
    info = {
        "sampling_rate": fs,
        "num_samples": len(signal),
        "duration_sec": duration,
        "num_channels_orig": num_channels,
        "original_shape": original_shape,
        "nyquist_freq": fs / 2.0,
        "rms_val": float(np.sqrt(np.mean(signal**2))),
        "peak_val": float(np.max(np.abs(signal))) if len(signal) > 0 else 0.0,
    }

    return signal, fs, info


def normalize_signal(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak normalizes a discrete signal safely without clipping.
    x_norm[n] = (x[n] / max|x[n]|) * target_peak
    """
    if len(signal) == 0:
        return signal
    # Remove DC bias first
    signal = signal - np.mean(signal)
    max_val = np.max(np.abs(signal))
    if max_val > 1e-9:
        return (signal / max_val) * target_peak
    return signal


def generate_synthetic_signal(
    signal_type: str = "Vocal / Speech Formant",
    duration_sec: float = 2.0,
    fs: int = 16000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates deterministic academic test signals for educational demonstration.
    
    Returns:
        t (np.ndarray): Time vector in seconds.
        x_clean (np.ndarray): Discrete clean audio samples x[n].
        metadata (dict): Signal properties and mathematical description.
    """
    N = int(duration_sec * fs)
    t = np.arange(N) / float(fs)
    
    if signal_type == "Vocal / Speech Formant":
        # Simulates vowel sound /a/ with fundamental pitch f0 and vocal tract formants
        f0 = 130.0  # Fundamental frequency (Hz)
        # Formant frequencies for vowel /a/ (F1, F2, F3)
        F1, BW1 = 730.0, 80.0
        F2, BW2 = 1090.0, 100.0
        F3, BW3 = 2440.0, 120.0
        
        # Glottal pulse excitation source: harmonic series with 1/k decay
        glottal = np.zeros_like(t)
        for k in range(1, 35):
            freq = k * f0
            if freq < fs / 2.0:
                # Formant resonance shaping (Lorentzian acoustic filters)
                gain1 = np.exp(-((freq - F1) ** 2) / (2 * (BW1**2)))
                gain2 = 0.6 * np.exp(-((freq - F2) ** 2) / (2 * (BW2**2)))
                gain3 = 0.3 * np.exp(-((freq - F3) ** 2) / (2 * (BW3**2)))
                formant_gain = 0.15 + gain1 + gain2 + gain3
                glottal += (formant_gain / k) * np.sin(2 * np.pi * freq * t)
        
        # Apply smooth speech envelope (attack, sustain, decay)
        envelope = np.sin(np.pi * t / duration_sec) ** 0.5
        x_clean = glottal * envelope
        description = "Synthetic Speech Formant Simulation (Vowel /a/ with F0=130Hz, F1=730Hz, F2=1090Hz, F3=2440Hz)"

    elif signal_type == "Multi-Harmonic Acoustic Tone":
        # Musical chord / instrument with fundamental (A4 = 440 Hz) and rich natural harmonics
        f0 = 440.0
        harmonics = [
            (1 * f0, 1.0, 0.0),       # Fundamental A4
            (2 * f0, 0.5, np.pi/4),   # 2nd harmonic 880 Hz
            (3 * f0, 0.25, np.pi/2),  # 3rd harmonic 1320 Hz
            (4 * f0, 0.12, np.pi/3),  # 4th harmonic 1760 Hz
            (5 * f0, 0.06, 0.0),      # 5th harmonic 2200 Hz
        ]
        x_clean = np.zeros_like(t)
        for freq, amp, phase in harmonics:
            if freq < fs / 2.0:
                x_clean += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Tremolo / vibrato subtle modulation for realistic acoustic warmth
        vibrato = 1.0 + 0.05 * np.sin(2 * np.pi * 5.0 * t)
        decay_envelope = np.exp(-1.2 * t / duration_sec)
        x_clean = x_clean * vibrato * decay_envelope
        description = "Multi-Harmonic Acoustic Tone (Fundamental 440Hz + 880Hz + 1320Hz + 1760Hz Harmonics)"

    elif signal_type == "Linear Chirp (Frequency Sweep)":
        # Frequency sweep from 150 Hz to 3500 Hz
        f_start, f_end = 150.0, 3500.0
        # Phase integral: \phi(t) = 2\pi (f_start * t + (f_end - f_start)/(2*T) * t^2)
        k = (f_end - f_start) / duration_sec
        phase = 2 * np.pi * (f_start * t + 0.5 * k * (t**2))
        x_clean = np.sin(phase)
        description = f"Linear Frequency Chirp ({f_start} Hz -> {f_end} Hz over {duration_sec}s)"

    elif signal_type == "Dual-Tone Multi-Frequency (DTMF)":
        # Low group (770 Hz) + High group (1336 Hz) = Key '5'
        f_low = 770.0
        f_high = 1336.0
        x_clean = 0.5 * np.sin(2 * np.pi * f_low * t) + 0.5 * np.sin(2 * np.pi * f_high * t)
        description = f"DTMF Dual Tone ({f_low} Hz + {f_high} Hz)"

    else:
        # Default simple tone
        x_clean = np.sin(2 * np.pi * 500.0 * t)
        description = "Pure 500 Hz Sinusoidal Tone"

    x_clean = normalize_signal(x_clean)
    metadata = {
        "signal_type": signal_type,
        "duration_sec": duration_sec,
        "fs": fs,
        "num_samples": N,
        "description": description,
    }
    return t, x_clean, metadata


def inject_noise(
    x_clean: np.ndarray,
    noise_type: str = "50 Hz Powerline Hum",
    snr_db: float = 10.0,
    fs: int = 16000,
    custom_whine_freq: float = 3500.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Injects mathematically defined noise into a clean signal at an exact target SNR.
    
    Returns:
        x_noisy (np.ndarray): Noisy composite signal x[n] = x_clean[n] + v[n].
        noise (np.ndarray): The isolated noise vector v[n].
        info (dict): True SNR_in, noise power, description.
    """
    N = len(x_clean)
    t = np.arange(N) / float(fs)
    
    # Calculate Clean Signal Power: P_s = (1/N) \sum x_clean[n]^2
    p_signal = np.mean(x_clean ** 2)
    if p_signal < 1e-12:
        p_signal = 1e-12

    # Desired Noise Power for given SNR: P_noise = P_signal / 10^(SNR/10)
    target_p_noise = p_signal / (10.0 ** (snr_db / 10.0))

    if noise_type == "50 Hz Powerline Hum":
        # 50 Hz base hum + 100 Hz 2nd harmonic + 150 Hz 3rd harmonic
        raw_noise = (
            1.0 * np.sin(2 * np.pi * 50.0 * t) +
            0.5 * np.sin(2 * np.pi * 100.0 * t + 0.3) +
            0.25 * np.sin(2 * np.pi * 150.0 * t + 0.7)
        )
        desc = "Powerline Interference (50 Hz fundamental + 100 Hz, 150 Hz harmonics)"

    elif noise_type == "60 Hz Powerline Hum":
        # 60 Hz base hum + 120 Hz + 180 Hz harmonics
        raw_noise = (
            1.0 * np.sin(2 * np.pi * 60.0 * t) +
            0.5 * np.sin(2 * np.pi * 120.0 * t + 0.3) +
            0.25 * np.sin(2 * np.pi * 180.0 * t + 0.7)
        )
        desc = "Powerline Interference (60 Hz fundamental + 120 Hz, 180 Hz harmonics)"

    elif noise_type == "High-Frequency Whine (3.5 kHz)":
        # High-pitch single frequency whine interference
        f_whine = min(custom_whine_freq, (fs / 2.0) - 200.0)
        raw_noise = np.sin(2 * np.pi * f_whine * t) + 0.3 * np.sin(2 * np.pi * (f_whine + 50) * t)
        desc = f"High-Pitch Whine / Electronic Clock Interference ({f_whine:.0f} Hz)"

    elif noise_type == "White Gaussian Noise (AWGN)":
        # Broadband zero-mean Gaussian noise N(0, \sigma^2)
        np.random.seed(42)  # Seed for reproducible evaluation
        raw_noise = np.random.normal(0.0, 1.0, N)
        desc = "Additive White Gaussian Noise (AWGN) - Flat Power Spectral Density"

    elif noise_type == "Combined Hum + Whine + AWGN":
        np.random.seed(42)
        awgn = np.random.normal(0.0, 0.5, N)
        hum = 0.8 * np.sin(2 * np.pi * 50.0 * t)
        f_whine = min(custom_whine_freq, (fs / 2.0) - 200.0)
        whine = 0.6 * np.sin(2 * np.pi * f_whine * t)
        raw_noise = hum + whine + awgn
        desc = f"Combined Interference (50 Hz Hum + {f_whine:.0f} Hz Whine + AWGN)"

    else:
        np.random.seed(42)
        raw_noise = np.random.normal(0.0, 1.0, N)
        desc = "Random Noise"

    # Scale raw_noise to match target_p_noise exactly
    raw_p_noise = np.mean(raw_noise ** 2)
    if raw_p_noise > 1e-12:
        scale = np.sqrt(target_p_noise / raw_p_noise)
        noise = raw_noise * scale
    else:
        noise = raw_noise

    # Form composite noisy signal x[n] = x_clean[n] + v[n]
    x_noisy = x_clean + noise

    # Calculate actual achieved Input SNR
    actual_p_noise = np.mean(noise ** 2)
    actual_snr_db = 10.0 * np.log10(p_signal / actual_p_noise) if actual_p_noise > 1e-12 else 99.0

    info = {
        "noise_type": noise_type,
        "target_snr_db": snr_db,
        "actual_snr_db": actual_snr_db,
        "signal_power": float(p_signal),
        "noise_power": float(actual_p_noise),
        "description": desc,
    }
    return x_noisy, noise, info


def audio_to_wav_bytes(signal: np.ndarray, fs: int) -> bytes:
    """
    Encodes discrete float array into standard 16-bit PCM WAV bytes in-memory.
    """
    # Clip safely to [-1.0, 1.0]
    clipped = np.clip(signal, -1.0, 1.0)
    # Convert to 16-bit signed integer PCM
    pcm16 = (clipped * 32767.0).astype(np.int16)
    
    buffer = io.BytesIO()
    wavfile.write(buffer, fs, pcm16)
    return buffer.getvalue()


def create_bundled_sample_files(target_dir: str):
    """
    Creates ready-to-use WAV sample files in the specified directory.
    """
    import os
    os.makedirs(target_dir, exist_ok=True)
    fs = 16000

    samples = [
        ("1_vocal_speech_clean.wav", "Vocal / Speech Formant", None, 0.0),
        ("2_vocal_speech_with_50hz_hum.wav", "Vocal / Speech Formant", "50 Hz Powerline Hum", 8.0),
        ("3_vocal_speech_with_3500hz_whine.wav", "Vocal / Speech Formant", "High-Frequency Whine (3.5 kHz)", 10.0),
        ("4_vocal_speech_with_gaussian_noise.wav", "Vocal / Speech Formant", "White Gaussian Noise (AWGN)", 6.0),
        ("5_music_harmonics_with_hum.wav", "Multi-Harmonic Acoustic Tone", "50 Hz Powerline Hum", 10.0),
        ("6_chirp_sweep_with_interference.wav", "Linear Chirp (Frequency Sweep)", "High-Frequency Whine (3.5 kHz)", 12.0),
    ]

    for fname, sig_type, noise_type, snr in samples:
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            _, x_c, _ = generate_synthetic_signal(sig_type, duration_sec=2.5, fs=fs)
            if noise_type:
                x_out, _, _ = inject_noise(x_c, noise_type=noise_type, snr_db=snr, fs=fs)
            else:
                x_out = x_c
            clipped = np.clip(x_out, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16)
            wavfile.write(fpath, fs, pcm)

