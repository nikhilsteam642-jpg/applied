"""
Generate pre-bundled WAV sample files for testing and viva demonstration.
Creates sample files in c:/projects/applied/samples/
"""

import os
import numpy as np
from scipy.io import wavfile

from modules.audio_processing import generate_synthetic_signal, inject_noise

def generate_sample_suite():
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    fs = 16000

    # 1. Clean Vocal Speech
    _, speech_clean, _ = generate_synthetic_signal("Vocal / Speech Formant", duration_sec=2.5, fs=fs)
    save_wav(os.path.join(samples_dir, "1_vocal_speech_clean.wav"), speech_clean, fs)

    # 2. Vocal Speech + 50 Hz Powerline Hum
    speech_hum, _, _ = inject_noise(speech_clean, "50 Hz Powerline Hum", snr_db=8.0, fs=fs)
    save_wav(os.path.join(samples_dir, "2_vocal_speech_with_50hz_hum.wav"), speech_hum, fs)

    # 3. Vocal Speech + High-Pitch Whine (3.5 kHz)
    speech_whine, _, _ = inject_noise(speech_clean, "High-Frequency Whine (3.5 kHz)", snr_db=10.0, fs=fs, custom_whine_freq=3500.0)
    save_wav(os.path.join(samples_dir, "3_vocal_speech_with_3500hz_whine.wav"), speech_whine, fs)

    # 4. Vocal Speech + White Gaussian Noise (AWGN)
    speech_awgn, _, _ = inject_noise(speech_clean, "White Gaussian Noise (AWGN)", snr_db=6.0, fs=fs)
    save_wav(os.path.join(samples_dir, "4_vocal_speech_with_gaussian_noise.wav"), speech_awgn, fs)

    # 5. Acoustic Musical Chord (Harmonics) + Powerline Hum
    _, music_clean, _ = generate_synthetic_signal("Multi-Harmonic Acoustic Tone", duration_sec=2.5, fs=fs)
    music_hum, _, _ = inject_noise(music_clean, "50 Hz Powerline Hum", snr_db=10.0, fs=fs)
    save_wav(os.path.join(samples_dir, "5_music_harmonics_with_hum.wav"), music_hum, fs)

    # 6. Frequency Chirp Sweep + High Pitch Interference
    _, chirp_clean, _ = generate_synthetic_signal("Linear Chirp (Frequency Sweep)", duration_sec=2.5, fs=fs)
    chirp_noisy, _, _ = inject_noise(chirp_clean, "High-Frequency Whine (3.5 kHz)", snr_db=12.0, fs=fs, custom_whine_freq=4000.0)
    save_wav(os.path.join(samples_dir, "6_chirp_sweep_with_interference.wav"), chirp_noisy, fs)

    print(f"Generated 6 WAV audio samples in: {samples_dir}")

def save_wav(filepath, signal, fs):
    clipped = np.clip(signal, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    wavfile.write(filepath, fs, pcm16)

if __name__ == "__main__":
    generate_sample_suite()
