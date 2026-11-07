"""
Audio Signal Enhancement Using Z-Transform Based Digital Filtering
Applied Mathematics & Digital Signal Processing Capstone Project

Author / Student: Siddhartha (Applied Mathematics Capstone)
Architecture: Streamlit + NumPy + SciPy + Matplotlib + SoundFile
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import time

# Import modular project packages
from modules.audio_processing import (
    load_audio_from_file,
    generate_synthetic_signal,
    inject_noise,
    audio_to_wav_bytes,
    normalize_signal,
    create_bundled_sample_files
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

# ---------------------------------------------------------
# Page Configuration & Custom Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Audio Signal Enhancement | Z-Transform DSP",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern engineering / scientific cards & badges
st.markdown("""
<style>
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0369A1 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC !important;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 14.5px;
        color: #7DD3FC !important;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .badge {
        display: inline-block;
        font-size: 11.5px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-teal {
        background: rgba(45, 212, 191, 0.15);
        color: #2DD4BF !important;
        border: 1px solid rgba(45, 212, 191, 0.3);
    }
    
    /* Section Cards */
    .metric-card {
        background: #1C2541;
        border: 1px solid #2A385B;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #00B4D8;
    }
    .metric-label {
        font-size: 11.5px;
        color: #8D99AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Mathematical Theory Box */
    .math-box {
        background: #1C2541;
        border-left: 4px solid #00B4D8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        color: #E0FBFC;
    }
    .math-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #00B4D8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "audio_source" not in st.session_state:
    st.session_state.audio_source = "preset"
if "x_raw" not in st.session_state:
    # Initialize with default synthetic speech
    t_init, x_init, meta_init = generate_synthetic_signal("Vocal / Speech Formant", duration_sec=2.0, fs=16000)
    x_noisy_init, noise_init, n_info_init = inject_noise(x_init, "50 Hz Powerline Hum", snr_db=10.0, fs=16000)
    st.session_state.fs = 16000
    st.session_state.t = t_init
    st.session_state.x_clean = x_init
    st.session_state.x_noisy = x_noisy_init
    st.session_state.noise = noise_init
    st.session_state.noise_info = n_info_init
    st.session_state.signal_meta = meta_init
    st.session_state.has_ground_truth = True
    st.session_state.filename = "synthetic_vocal_50hz_hum.wav"


# ---------------------------------------------------------
# Sidebar Navigation & Global Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ Navigation & Workflow")
    selected_page = st.radio(
        "Select Pipeline Stage:",
        [
            "1. Architecture & Math Foundations",
            "2. Audio Acquisition & Preprocessing",
            "3. Time & Spectral Domain Analysis",
            "4. Z-Transform Modeling",
            "5. Digital Filter Design (Z-Domain)",
            "6. Audio Enhancement & Filtering",
            "7. Performance Evaluation",
            "8. Capstone Presentation & Viva"
        ],
        index=1
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Audio Status")
    curr_fs = st.session_state.get("fs", 16000)
    curr_samples = len(st.session_state.get("x_noisy", []))
    curr_dur = curr_samples / float(curr_fs) if curr_fs > 0 else 0.0
    
    st.markdown(f"""
    - **Source**: `{st.session_state.get('filename', 'Audio')}`
    - **Sampling Rate ($f_s$)**: `{curr_fs} Hz`
    - **Nyquist ($f_{{nyq}}$)**: `{curr_fs / 2:.0f} Hz`
    - **Duration**: `{curr_dur:.2f} s` (`{curr_samples:,}` samples)
    - **Reference Signal**: `{'Available (Exact SNR)' if st.session_state.get('has_ground_truth') else 'Upload (Proxy Metrics)'}`
    """)


# ---------------------------------------------------------
# Header / Hero Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Audio Signal Enhancement Using Z-Transform Based Digital Filtering</div>
    <div class="hero-subtitle">Applied Mathematics & Digital Signal Processing Capstone Project</div>
    <div class="hero-badges">
        <span class="badge">Discrete-Time Signals x[n]</span>
        <span class="badge">Laurent Polynomial X(z) = Σ x[n] z⁻ⁿ</span>
        <span class="badge-teal">LCCDE Difference Equations</span>
        <span class="badge">BIBO Stability Theorem (|pₖ| &lt; 1)</span>
        <span class="badge-teal">FIR / IIR Butterworth & Chebyshev</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: SYSTEM ARCHITECTURE & MATHEMATICAL FOUNDATIONS
# =============================================================================
if selected_page == "1. Architecture & Math Foundations":
    st.markdown("## 🏛️ System Architecture & Mathematical Foundations")
    st.markdown("This capstone project translates continuous audio phenomena into discrete-time mathematical structures, applies Laurent polynomial Z-transform modeling, synthesizes stable digital filters in the complex Z-plane, and reconstructs clean acoustic waveforms.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🔄 End-to-End System Workflow")
        st.markdown("""
        ```mermaid
        graph TD
            A[Audio Input: Continuous / File] --> B[Preprocessing: Sampling & Normalization x_n]
            B --> C[Time-Domain Analysis: RMS, Peak, Energy]
            C --> D[Frequency-Domain FFT: X_k = Σ x_n e^-j2πkn/N]
            D --> E[Z-Transform Modeling: X_z = Σ x_n z^-n]
            E --> F[Z-Domain Filter Synthesis: H_z = B_z / A_z]
            F --> G[Pole-Zero Stability Verification: |p_k| < 1]
            G --> H[Digital Filtering Execution: lfilter / filtfilt]
            H --> I[Enhanced Audio Reconstruction: y_n]
            I --> J[Performance Evaluation: ΔSNR, MSE, Spectrograms]
        ```
        """)

    with col2:
        st.markdown("### 📐 Core Mathematical Equations")
        
        st.markdown("""
        <div class="math-box">
            <div class="math-title">1. Discrete-Time Sampling</div>
            Continuous audio $x_a(t)$ is sampled at rate $f_s = 1/T_s$:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"x[n] = x_a(n T_s), \quad n \in \{0, 1, \dots, N-1\}")

        st.markdown("""
        <div class="math-box">
            <div class="math-title">2. Discrete Fourier Transform (FFT)</div>
            Evaluates the spectral decomposition across discrete frequency bins:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi k n}{N}}, \quad f_k = \frac{k f_s}{N}")

        st.markdown("""
        <div class="math-box">
            <div class="math-title">3. Z-Transform Laurent Series</div>
            Maps the discrete-time sequence into the complex frequency $z$-plane ($z = r e^{j\omega}$):
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=0}^{N-1} x[n] z^{-n} = \frac{\sum_{n=0}^{N-1} x[n] z^{N-1-n}}{z^{N-1}}")

        st.markdown("""
        <div class="math-box">
            <div class="math-title">4. Digital Transfer Function & LCCDE</div>
            Rational transfer function in the Z-domain:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}} = K \frac{\prod_{i=1}^M (1 - q_i z^{-1})}{\prod_{k=1}^N (1 - p_k z^{-1})}")

        st.markdown("""
        <div class="math-box">
            <div class="math-title">5. BIBO Stability Theorem</div>
            A causal LTI digital filter is Bounded-Input Bounded-Output (BIBO) stable if and only if:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\max_{k} |p_k| < 1.0000 \quad (\text{All poles lie strictly inside the unit circle } |z|=1)")


# =============================================================================
# PAGE 2: AUDIO ACQUISITION & PREPROCESSING
# =============================================================================
elif selected_page == "2. Audio Acquisition & Preprocessing":
    st.markdown("## 🎙️ Audio Acquisition & Preprocessing")
    st.markdown("Acquire audio via **User File Upload** or use the **Academic Synthetic Signal & Calibrated Noise Generator** to establish a ground-truth clean reference for exact $\\Delta\\text{SNR}$ validation.")

    input_tab1, input_tab2, input_tab3 = st.tabs([
        "🔬 Academic Signal Generator (Clean + Noise)",
        "🎵 Pre-Bundled WAV Audio Samples",
        "📁 Upload Custom Audio File"
    ])

    with input_tab1:
        st.markdown("### Generate Clean Signal + Inject Calibrated Noise")
        col_gen1, col_gen2, col_gen3 = st.columns(3)

        with col_gen1:
            sig_type = st.selectbox(
                "Clean Acoustic Signal:",
                [
                    "Vocal / Speech Formant",
                    "Multi-Harmonic Acoustic Tone",
                    "Linear Chirp (Frequency Sweep)",
                    "Dual-Tone Multi-Frequency (DTMF)"
                ],
                index=0
            )
            sig_duration = st.slider("Signal Duration (seconds):", 1.0, 4.0, 2.0, 0.5)

        with col_gen2:
            noise_type = st.selectbox(
                "Interference / Noise Type:",
                [
                    "50 Hz Powerline Hum",
                    "60 Hz Powerline Hum",
                    "High-Frequency Whine (3.5 kHz)",
                    "White Gaussian Noise (AWGN)",
                    "Combined Hum + Whine + AWGN"
                ],
                index=0
            )
            fs_choice = st.selectbox("Sampling Frequency (fs):", [8000, 16000, 22050, 44100], index=1)

        with col_gen3:
            target_snr = st.slider("Input SNR (dB) Target:", 0.0, 30.0, 10.0, 1.0)
            whine_f = st.slider("Custom Whine Freq (Hz):", 1000.0, 6000.0, 3500.0, 100.0)

        if st.button("🚀 Generate & Load Synthetic Audio", type="primary"):
            t_gen, x_clean_gen, meta_gen = generate_synthetic_signal(sig_type, duration_sec=sig_duration, fs=fs_choice)
            x_noisy_gen, noise_gen, n_info_gen = inject_noise(
                x_clean_gen, noise_type=noise_type, snr_db=target_snr, fs=fs_choice, custom_whine_freq=whine_f
            )
            st.session_state.fs = fs_choice
            st.session_state.t = t_gen
            st.session_state.x_clean = x_clean_gen
            st.session_state.x_noisy = x_noisy_gen
            st.session_state.noise = noise_gen
            st.session_state.noise_info = n_info_gen
            st.session_state.signal_meta = meta_gen
            st.session_state.has_ground_truth = True
            st.session_state.filename = f"synthetic_{sig_type.lower().replace(' ', '_')}_{noise_type.lower()[:8]}.wav"
            st.success(f"Generated {sig_type} with {noise_type} at actual SNR: {n_info_gen['actual_snr_db']:.2f} dB!")

    with input_tab2:
        st.markdown("### 🎵 Select a Pre-Bundled Capstone Sample File")
        samples_dir = os.path.join(os.path.dirname(__file__), "samples")
        create_bundled_sample_files(samples_dir)

        bundled_samples = {
            "2_vocal_speech_with_50hz_hum.wav": ("Speech Formants + 50 Hz Powerline Hum", "Vocal / Speech Formant", "50 Hz Powerline Hum", 8.0),
            "3_vocal_speech_with_3500hz_whine.wav": ("Speech Formants + 3.5 kHz High-Pitch Whine", "Vocal / Speech Formant", "High-Frequency Whine (3.5 kHz)", 10.0),
            "4_vocal_speech_with_gaussian_noise.wav": ("Speech Formants + White Gaussian Noise", "Vocal / Speech Formant", "White Gaussian Noise (AWGN)", 6.0),
            "5_music_harmonics_with_hum.wav": ("Musical Acoustic Harmonics + 50 Hz Hum", "Multi-Harmonic Acoustic Tone", "50 Hz Powerline Hum", 10.0),
            "6_chirp_sweep_with_interference.wav": ("Chirp Frequency Sweep + 4 kHz Whine", "Linear Chirp (Frequency Sweep)", "High-Frequency Whine (3.5 kHz)", 12.0),
            "1_vocal_speech_clean.wav": ("Clean Vocal Speech Reference", "Vocal / Speech Formant", None, 0.0),
        }

        selected_sample_key = st.selectbox(
            "Choose a demo scenario:",
            list(bundled_samples.keys()),
            format_func=lambda k: f"📁 {k} — ({bundled_samples[k][0]})"
        )

        sample_desc, sig_t, noise_t, snr_v = bundled_samples[selected_sample_key]
        st.info(f"**Scenario Description:** {sample_desc}")

        if st.button("📥 Load Selected Sample File", type="primary"):
            sample_path = os.path.join(samples_dir, selected_sample_key)
            sig_s, fs_s, info_s = load_audio_from_file(sample_path)
            t_s = np.arange(len(sig_s)) / float(fs_s)
            
            # If known scenario, recover clean reference for exact metrics
            if noise_t is not None:
                _, x_clean_s, _ = generate_synthetic_signal(sig_t, duration_sec=len(sig_s)/fs_s, fs=fs_s)
                st.session_state.x_clean = x_clean_s
                st.session_state.has_ground_truth = True
            else:
                st.session_state.x_clean = sig_s.copy()
                st.session_state.has_ground_truth = True

            st.session_state.fs = fs_s
            st.session_state.t = t_s
            st.session_state.x_noisy = sig_s
            st.session_state.noise = None
            st.session_state.noise_info = None
            st.session_state.signal_meta = info_s
            st.session_state.filename = selected_sample_key
            st.success(f"Loaded '{selected_sample_key}' successfully!")

    with input_tab3:
        st.markdown("### Upload Your Own Audio File (WAV, MP3, OGG, FLAC)")
        uploaded_file = st.file_uploader("Choose an audio file from your computer:", type=["wav", "mp3", "ogg", "flac"])
        if uploaded_file is not None:
            try:
                sig_up, fs_up, info_up = load_audio_from_file(uploaded_file)
                t_up = np.arange(len(sig_up)) / float(fs_up)
                st.session_state.fs = fs_up
                st.session_state.t = t_up
                st.session_state.x_clean = None
                st.session_state.x_noisy = sig_up
                st.session_state.noise = None
                st.session_state.noise_info = None
                st.session_state.signal_meta = info_up
                st.session_state.has_ground_truth = False
                st.session_state.filename = uploaded_file.name
                st.success(f"Loaded '{uploaded_file.name}' ({len(sig_up):,} samples @ {fs_up} Hz). Mono-converted & normalized.")
            except Exception as e:
                st.error(f"Error loading audio file: {e}")

    # Display Current Audio Status & Players
    st.markdown("---")
    st.markdown("### 📊 Active Audio Signals & Playback")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.metric("Sampling Rate", f"{st.session_state.fs} Hz")
    with col_meta2:
        st.metric("Samples Count", f"{len(st.session_state.x_noisy):,}")
    with col_meta3:
        st.metric("Duration", f"{len(st.session_state.x_noisy) / st.session_state.fs:.2f} s")
    with col_meta4:
        st.metric("Nyquist Frequency", f"{st.session_state.fs / 2:.0f} Hz")

    col_play1, col_play2 = st.columns(2)
    with col_play1:
        st.markdown("#### 🔊 Input / Noisy Audio Signal $x[n]$")
        wav_noisy = audio_to_wav_bytes(st.session_state.x_noisy, st.session_state.fs)
        st.audio(wav_noisy, format="audio/wav")

    with col_play2:
        if st.session_state.get("has_ground_truth") and st.session_state.get("x_clean") is not None:
            st.markdown("#### 🎯 Ground Truth Clean Reference $x_{clean}[n]$")
            wav_clean = audio_to_wav_bytes(st.session_state.x_clean, st.session_state.fs)
            st.audio(wav_clean, format="audio/wav")
        else:
            st.markdown("#### ℹ️ Ground Truth Note")
            st.info("User-uploaded file loaded. Ground-truth reference is unavailable; evaluation will use rigorous spectral proxy metrics (stopband attenuation, out-of-band energy rejection).")

    # Waveform Plot
    fig_raw = plot_waveform_single(
        st.session_state.t, st.session_state.x_noisy,
        title=f"Discrete-Time Signal x[n] : {st.session_state.filename}"
    )
    st.pyplot(fig_raw)
    plt.close(fig_raw)


# =============================================================================
# PAGE 3: TIME & SPECTRAL DOMAIN ANALYSIS
# =============================================================================
elif selected_page == "3. Time & Spectral Domain Analysis":
    st.markdown("## 📊 Time & Spectral Domain Analysis")
    st.markdown("Discrete-time analysis and Discrete Fourier Transform (FFT) decomposing $x[n]$ into its harmonic basis components.")

    fs = st.session_state.fs
    x = st.session_state.x_noisy
    t = st.session_state.t

    # Compute FFT
    fft_res = compute_fft(x, fs=fs, window="hann")
    freqs = fft_res["freqs"]
    mag_db = fft_res["mag_db_normalized"]
    peaks = fft_res["dominant_peaks"]
    stats = fft_res["spectral_stats"]

    col_st1, col_st2, col_st3, col_st4 = st.columns(4)
    with col_st1:
        rms_val = float(np.sqrt(np.mean(x**2)))
        st.metric("RMS Amplitude", f"{rms_val:.4f}")
    with col_st2:
        peak_val = float(np.max(np.abs(x)))
        st.metric("Peak Amplitude", f"{peak_val:.4f}")
    with col_st3:
        st.metric("Spectral Centroid", f"{stats['spectral_centroid_hz']:.1f} Hz")
    with col_st4:
        st.metric("Spectral Bandwidth", f"{stats['spectral_bandwidth_hz']:.1f} Hz")

    # FFT Spectrum Plot
    st.markdown("### 📈 FFT Magnitude Spectrum $|X[k]|$")
    fig_fft = plot_fft_spectrum(freqs, mag_db, dominant_peaks=peaks, title="Single-Sided Discrete Fourier Spectrum (FFT)")
    st.pyplot(fig_fft)
    plt.close(fig_fft)

    # Dominant Peaks Table
    col_pk1, col_pk2 = st.columns([1, 1])
    with col_pk1:
        st.markdown("#### 🎯 Dominant Spectral Peaks")
        if peaks:
            peak_data = [{"Harmonic Rank": i+1, "Frequency (Hz)": f"{p['frequency_hz']:.1f}", "Magnitude (dB rel)": f"{p['magnitude_db_rel']:.2f}"} for i, p in enumerate(peaks)]
            st.table(peak_data)
        else:
            st.write("No distinct narrow peaks detected.")

    with col_pk2:
        st.markdown("#### 📐 Mathematical Basis: Discrete Fourier Transform")
        st.markdown("""
        <div class="math-box">
            The Discrete Fourier Transform (DFT) evaluates the Z-transform on $N$ equally spaced points on the unit circle $z = e^{j 2\pi k / N}$:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \cdot w[n] \cdot e^{-j \frac{2\pi k n}{N}}")
        st.markdown(f"""
        - **FFT Size ($N$)**: `{fft_res['n_fft']}` bins
        - **Frequency Resolution ($\Delta f$)**: `{fs / fft_res['n_fft']:.2f} Hz/bin`
        - **Spectral Flatness (Wiener Entropy)**: `{stats['spectral_flatness']:.4f}`
        - **85% Energy Spectral Rolloff**: `{stats['spectral_rolloff_85_hz']:.1f} Hz`
        """)


# =============================================================================
# PAGE 4: Z-TRANSFORM MODELING
# =============================================================================
elif selected_page == "4. Z-Transform Modeling":
    st.markdown("## 🌀 Z-Transform Mathematical Modeling")
    st.markdown("Comprehensive evaluation of the discrete audio sequence as a Laurent polynomial in the complex $z$-plane.")

    fs = st.session_state.fs
    x = st.session_state.x_noisy

    col_zcfg1, col_zcfg2 = st.columns([1, 2])
    with col_zcfg1:
        st.markdown("### ⚙️ Analysis Window Configuration")
        w_len = st.select_slider("Analysis Window Length N (samples):", options=[16, 32, 48, 64, 96, 128], value=32)
        start_ratio = st.slider("Window Position in Audio (%):", 0, 90, 30, 5) / 100.0
        x_win = extract_analysis_window(x, window_length=w_len, start_ratio=start_ratio)

    with col_zcfg2:
        st.markdown("### 📜 Discrete Sequence Laurent Polynomial")
        poly_latex = format_polynomial_latex(x_win, max_terms=7)
        st.latex(poly_latex)
        st.markdown("""
        <div class="math-box">
            For a finite-length discrete sequence $x[n]$, the unilateral/bilateral Z-transform expands into a finite Laurent polynomial with Region of Convergence (ROC) covering the entire complex plane except $z=0$.
        </div>
        """, unsafe_allow_html=True)

    # Compute roots (zeros) of the window polynomial
    z_res = compute_signal_zeros(x_win)
    zeros = z_res["zeros"]
    poles = z_res["poles"]

    col_z1, col_z2 = st.columns([1, 1])

    with col_z1:
        st.markdown("### ⭕ Signal Complex Z-Plane Zeros")
        st.markdown(f"Extracting roots of $X(z) = 0$ yields **{len(zeros)} complex zeros** representing spectral attenuation directions.")
        fig_pz = plot_pole_zero_diagram(zeros, poles[:1], is_stable=True, title=f"Signal Zeros in Complex Plane (N={w_len})")
        st.pyplot(fig_pz)
        plt.close(fig_pz)

    with col_z2:
        st.markdown("### 🌐 Complex Z-Plane Manifold $|X(z)|$")
        st.markdown("Evaluating Laurent polynomial $|X(r e^{j\theta})|$ over $r \in [0.5, 1.5], \theta \in [-\pi, \pi]$.")
        surface_data = evaluate_z_plane_surface(x_win, r_min=0.5, r_max=1.5, n_r=35, n_theta=100)
        fig_surf = plot_z_plane_surface_2d(
            surface_data["r_vals"], surface_data["theta_vals"], surface_data["mag_surface_db"]
        )
        st.pyplot(fig_surf)
        plt.close(fig_surf)

    st.markdown("""
    <div class="math-box">
        <div class="math-title">Unit Circle DTFT Cross-Section Equivalence</div>
        When $z = e^{j\omega}$ (i.e. on the red dashed unit circle $|z|=1$), the Z-transform magnitude $|X(z)|$ simplifies exactly to the Discrete-Time Fourier Transform (DTFT) $|X(e^{j\omega})|$. The zeros near the unit circle create spectral dips/notches, while poles outside would indicate unbounded energy growth.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5: DIGITAL FILTER DESIGN (Z-DOMAIN)
# =============================================================================
elif selected_page == "5. Digital Filter Design (Z-Domain)":
    st.markdown("## 🎛️ Digital Filter Design in the Z-Domain")
    st.markdown("Synthesize FIR or IIR digital filters by placing poles and zeros in the complex $z$-plane to shape the frequency response $H(z)$.")

    fs = st.session_state.fs
    nyquist = fs / 2.0

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        f_family = st.selectbox(
            "Filter Architecture:",
            [
                "Butterworth (IIR)",
                "Chebyshev Type I (IIR)",
                "Chebyshev Type II (IIR)",
                "Elliptic (Cauer) (IIR)",
                "FIR Windowed (Hamming)",
                "FIR Windowed (Blackman)",
                "Digital Notch Filter (IIR)"
            ],
            index=0
        )

    with col_f2:
        if f_family == "Digital Notch Filter (IIR)":
            f_type = "Notch (Interference Removal)"
            st.selectbox("Filter Passband Type:", ["Notch (Interference Removal)"], disabled=True)
        else:
            f_type = st.selectbox(
                "Filter Passband Type:",
                ["Low-pass", "High-pass", "Band-pass", "Band-stop"],
                index=0
            )

    with col_f3:
        if f_family.startswith("FIR"):
            f_order = st.slider("FIR Filter Order (Complexity):", 4, 60, 20, 2)
        elif f_family == "Digital Notch Filter (IIR)":
            f_order = 2
            st.slider("IIR Order:", 2, 2, 2, disabled=True)
        else:
            f_order = st.slider("IIR Filter Order (Poles):", 1, 10, 4, 1)

    # Cutoff frequencies configuration
    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        if f_type in ["Low-pass", "High-pass", "Notch (Interference Removal)"]:
            default_fc = min(1000.0, nyquist * 0.4)
            if "50 Hz" in st.session_state.get("filename", ""):
                default_fc = 80.0 if f_type == "High-pass" else 50.0
            fc1 = st.slider("Cutoff / Center Frequency (Hz):", 10.0, float(nyquist - 20.0), float(default_fc), 10.0)
            fc2 = None
        else:
            fc1 = st.slider("Lower Cutoff Frequency f1 (Hz):", 20.0, float(nyquist - 100.0), 300.0, 10.0)
            fc2 = st.slider("Upper Cutoff Frequency f2 (Hz):", float(fc1 + 20.0), float(nyquist - 20.0), min(3000.0, float(nyquist - 50.0)), 10.0)

    with col_fc2:
        if f_family == "Digital Notch Filter (IIR)":
            notch_q = st.slider("Notch Quality Factor Q (Sharpness):", 5.0, 100.0, 30.0, 5.0)
            rp, rs = 1.0, 40.0
        elif "Chebyshev Type I" in f_family or "Elliptic" in f_family:
            rp = st.slider("Passband Ripple (dB):", 0.1, 5.0, 1.0, 0.1)
            rs = 40.0
            notch_q = 30.0
        elif "Chebyshev Type II" in f_family:
            rs = st.slider("Stopband Rejection (dB):", 20.0, 80.0, 40.0, 5.0)
            rp = 1.0
            notch_q = 30.0
        else:
            rp, rs, notch_q = 1.0, 40.0, 30.0
            st.info("Butterworth / FIR filters provide maximally flat passbands without configurable ripple parameters.")

    # Design the digital filter
    filter_design_res = design_digital_filter(
        filter_family=f_family,
        filter_type=f_type,
        cutoff_freq=fc1,
        cutoff_freq_high=fc2,
        filter_order=f_order,
        fs=fs,
        rp=rp,
        rs=rs,
        notch_q=notch_q
    )
    # Store in session state for filtering stage
    st.session_state.filter_design = filter_design_res

    st.markdown("---")
    st.markdown(f"### 📋 Synthesized Transfer Function: `{filter_design_res['family_desc']}`")

    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown("**Transfer Function $H(z)$:**")
        st.latex(filter_design_res["hz_latex"])
        st.markdown("**Difference Equation $y[n]$:**")
        st.latex(filter_design_res["diff_eq_latex"])

    with col_h2:
        st.markdown("**Stability & Pole-Zero Summary:**")
        stab_color = "green" if filter_design_res["is_stable"] else "red"
        st.markdown(f"""
        - **BIBO Stability**: :{stab_color}[{'STABLE' if filter_design_res['is_stable'] else 'UNSTABLE'}]
        - **Max Pole Radius $\max |p_k|$**: `{filter_design_res['max_pole_mag']:.4f}`
        - **Stability Margin ($1 - \max |p_k|$)**: `{filter_design_res['stability_margin']:.4f}`
        - **Number of Zeros**: `{filter_design_res['num_zeros']}`
        - **Number of Poles**: `{filter_design_res['num_poles']}`
        """)

    # Plots: Pole-Zero Plot + Frequency Response Plot
    col_p1, col_p2 = st.columns([1, 1.2])
    with col_p1:
        fig_fpz = plot_pole_zero_diagram(
            filter_design_res["zeros"],
            filter_design_res["poles"],
            is_stable=filter_design_res["is_stable"],
            title=f"Filter Z-Plane Poles & Zeros"
        )
        st.pyplot(fig_fpz)
        plt.close(fig_fpz)

    with col_p2:
        fig_fresp = plot_frequency_response(
            filter_design_res["freqs"],
            filter_design_res["mag_db"],
            filter_design_res["phase_deg"],
            cutoff_freq=fc1,
            cutoff_freq_high=fc2
        )
        st.pyplot(fig_fresp)
        plt.close(fig_fresp)


# =============================================================================
# PAGE 6: AUDIO ENHANCEMENT & FILTERING
# =============================================================================
elif selected_page == "6. Audio Enhancement & Filtering":
    st.markdown("## ⚡ Audio Enhancement & Filter Application")
    st.markdown("Execute the synthesized digital filter on the discrete audio samples $x[n]$ to produce the enhanced signal $y[n]$.")

    fs = st.session_state.fs
    x = st.session_state.x_noisy
    t = st.session_state.t

    # Ensure filter is designed
    if "filter_design" not in st.session_state:
        st.session_state.filter_design = design_digital_filter(
            filter_family="Butterworth (IIR)", filter_type="Low-pass", cutoff_freq=1200.0, filter_order=4, fs=fs
        )
    f_info = st.session_state.filter_design

    col_cfg1, col_cfg2 = st.columns([1, 1])
    with col_cfg1:
        filt_method = st.radio(
            "Digital Filtering Algorithm:",
            [
                "filtfilt (Zero-Phase Forward-Backward Filtering - Recommended)",
                "lfilter (Causal Direct Form II Transposed Difference Equation)"
            ],
            index=0
        )
        method_key = "filtfilt" if "filtfilt" in filt_method else "lfilter"

    with col_cfg2:
        st.markdown("### 🎛️ Active Filter Parameters")
        st.markdown(f"""
        - **Architecture**: `{f_info['filter_family']}`
        - **Type**: `{f_info['filter_type']}`
        - **Cutoff**: `{f_info['cutoff_freq']:.1f} Hz` {f"to {f_info['cutoff_freq_high']:.1f} Hz" if f_info.get('cutoff_freq_high') else ''}
        - **Stability**: `{'BIBO Stable' if f_info['is_stable'] else 'Unstable'}`
        """)

    # Apply digital filter
    y_enh, residual, exec_info = apply_digital_filter(
        x, b=f_info["b"], a=f_info["a"], method=method_key, preserve_scale=True
    )
    st.session_state.y_enh = y_enh
    st.session_state.residual = residual
    st.session_state.exec_info = exec_info

    st.markdown("---")
    st.markdown("### 🔊 In-Browser Audio Playback & Enhancement Comparison")

    col_au1, col_au2, col_au3 = st.columns(3)
    with col_au1:
        st.markdown("#### 1. Input / Noisy Audio $x[n]$")
        wav_orig = audio_to_wav_bytes(x, fs)
        st.audio(wav_orig, format="audio/wav")

    with col_au2:
        st.markdown("#### 2. Enhanced Filtered Audio $y[n]$")
        wav_enh = audio_to_wav_bytes(y_enh, fs)
        st.audio(wav_enh, format="audio/wav")

    with col_au3:
        st.markdown("#### 3. Removed Noise / Residual $e[n]$")
        wav_res = audio_to_wav_bytes(residual, fs)
        st.audio(wav_res, format="audio/wav")

    st.download_button(
        label="💾 Download Enhanced Audio WAV File",
        data=wav_enh,
        file_name=f"enhanced_{st.session_state.get('filename', 'audio.wav')}",
        mime="audio/wav",
        type="primary"
    )

    # Waveform Comparison Plots
    st.markdown("---")
    st.markdown("### 📊 Time-Domain Waveform Comparison: $x[n]$ vs $y[n]$ vs $e[n]$")
    fig_wcmp = plot_waveform_comparison(t, x, y_enh)
    st.pyplot(fig_wcmp)
    plt.close(fig_wcmp)

    # Dual Spectrograms
    st.markdown("### 🌈 Time-Frequency Spectrogram Comparison")
    st.markdown("Notice how unwanted noise bands (e.g. 50Hz hum harmonic lines or high-frequency hiss) are eliminated in the bottom enhanced spectrogram.")
    fig_spec = plot_dual_spectrogram(x, y_enh, fs)
    st.pyplot(fig_spec)
    plt.close(fig_spec)


# =============================================================================
# PAGE 7: PERFORMANCE EVALUATION & QUALITY METRICS
# =============================================================================
elif selected_page == "7. Performance Evaluation":
    st.markdown("## 📈 Performance Evaluation & Audio Quality Analysis")
    st.markdown("Comprehensive mathematical metrics assessing signal energy, RMS power, noise reduction, and spectral attenuation.")

    fs = st.session_state.fs
    x = st.session_state.x_noisy
    x_clean = st.session_state.get("x_clean")
    f_info = st.session_state.get("filter_design", design_digital_filter(fs=fs))
    
    # Ensure enhanced audio exists
    if "y_enh" not in st.session_state:
        y_enh, residual, _ = apply_digital_filter(x, f_info["b"], f_info["a"])
        st.session_state.y_enh = y_enh
        st.session_state.residual = residual
    else:
        y_enh = st.session_state.y_enh
        residual = st.session_state.residual

    # Evaluate metrics
    perf = evaluate_audio_performance(
        x_orig=x,
        y_enh=y_enh,
        x_clean=x_clean,
        fs=fs,
        filter_type=f_info["filter_type"],
        cutoff_freq=f_info["cutoff_freq"],
        cutoff_freq_high=f_info.get("cutoff_freq_high")
    )

    # Display Metrics Cards
    if perf["has_ground_truth"]:
        st.markdown("### 🎯 Ground-Truth SNR & Error Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Input SNR (dB)", f"{perf['snr_in_db']:.2f} dB")
        with col_m2:
            st.metric("Enhanced SNR (dB)", f"{perf['snr_out_db']:.2f} dB")
        with col_m3:
            delta_color = "normal" if perf['delta_snr_db'] >= 0 else "inverse"
            st.metric("Δ SNR Improvement", f"{perf['delta_snr_db']:+.2f} dB", delta=f"{perf['delta_snr_db']:+.2f} dB")
        with col_m4:
            st.metric("MSE Reduction", f"{perf['mse_reduction_pct']:.1f} %")
    else:
        st.markdown("### 📊 Spectral Proxy Metrics (Reference-Free)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Stopband Attenuation", f"{perf['stopband_attenuation_db']:.2f} dB")
        with col_m2:
            st.metric("Energy Ratio (Out/In)", f"{perf['energy_ratio_db']:.2f} dB")
        with col_m3:
            st.metric("Residual RMS", f"{perf['residual_rms']:.4f}")
        with col_m4:
            st.metric("Enhanced Crest Factor", f"{perf['crest_enh']:.2f}")

    # General Signal Energy Statistics
    st.markdown("---")
    st.markdown("### 📐 Detailed Signal Statistics Table")
    stat_rows = [
        {"Metric": "RMS Amplitude", "Original x[n]": f"{perf['rms_orig']:.4f}", "Enhanced y[n]": f"{perf['rms_enh']:.4f}", "Unit": "Linear"},
        {"Metric": "Peak Amplitude", "Original x[n]": f"{perf['peak_orig']:.4f}", "Enhanced y[n]": f"{perf['peak_enh']:.4f}", "Unit": "Linear [-1, 1]"},
        {"Metric": "Crest Factor (Peak / RMS)", "Original x[n]": f"{perf['crest_orig']:.2f}", "Enhanced y[n]": f"{perf['crest_enh']:.2f}", "Unit": "Ratio"},
        {"Metric": "Total Signal Energy Σ x²[n]", "Original x[n]": f"{perf['energy_orig']:.2f}", "Enhanced y[n]": f"{perf['energy_enh']:.2f}", "Unit": "Energy (a.u.)"},
        {"Metric": "Stopband Rejection", "Original x[n]": "0.00 dB (Baseline)", "Enhanced y[n]": f"{perf['stopband_attenuation_db']:.2f} dB", "Unit": f"{perf['stopband_desc']}"}
    ]
    st.table(stat_rows)

    # FFT Spectral Overlay Plot
    st.markdown("### 📊 Spectral Comparison: Original vs Enhanced FFT Spectrum")
    fft_orig = compute_fft(x, fs)
    fft_enh = compute_fft(y_enh, fs)
    cutoffs = [f_info["cutoff_freq"]]
    if f_info.get("cutoff_freq_high"):
        cutoffs.append(f_info["cutoff_freq_high"])

    fig_ovl = plot_fft_comparison(fft_orig["freqs"], fft_orig["mag_db_normalized"], fft_enh["mag_db_normalized"], cutoff_freqs=cutoffs)
    st.pyplot(fig_ovl)
    plt.close(fig_ovl)


# =============================================================================
# PAGE 8: CAPSTONE PRESENTATION & VIVA SUMMARY
# =============================================================================
elif selected_page == "8. Capstone Presentation & Viva":
    st.markdown("## 🎓 Capstone Presentation & Viva Defense Guide")
    st.markdown("This guide maps the software implementation directly to the four academic modules for presentation before the evaluation panel.")

    tab_v1, tab_v2, tab_v3 = st.tabs(["📑 Module Mapping to Presentation PPT", "🎙️ Viva Step-by-Step Demo Script", "💡 Expected Evaluator Questions & Answers"])

    with tab_v1:
        st.markdown("""
        ### 📌 4-Module Academic Mapping

        | Module | Title | Covered Concepts & Implementation | Key Result / Deliverable |
        | :--- | :--- | :--- | :--- |
        | **Module 1** | **Audio Analysis & Z-Transform Modeling** | Discrete sampling $x[n]$, FFT decomposition $X[k]$, Laurent series $X(z) = \sum x[n] z^{-n}$, polynomial zeros, 2D complex manifold & unit circle DTFT equivalence. | Discrete representation, harmonic detection, complex Z-plane representation. |
        | **Module 2** | **Digital Filter Design (Z-Domain)** | Transfer function $H(z) = B(z)/A(z)$, FIR (Hamming/Hann) & IIR (Butterworth, Chebyshev I & II, Notch), pole-zero extraction, BIBO stability theorem ($\max \|p_k\| < 1$). | Verified stable filter coefficients $b, a$, frequency & phase response curves. |
        | **Module 3** | **Filter Implementation & Audio Processing** | Causal Direct Form II difference equation (`lfilter`) vs Zero-Phase forward-backward (`filtfilt`), sample-by-sample processing, WAV reconstruction. | Audio enhancement, noise separation $e[n] = x[n] - y[n]$, real-time playable audio. |
        | **Module 4** | **Performance Evaluation & Quality Analysis** | Time waveforms, spectral overlays, dual spectrograms, $\Delta\text{SNR}$ improvement, stopband attenuation (dB), RMS, MSE. | Quantitative noise reduction metrics ($\Delta\text{SNR} > +8\text{ dB}$, attenuation $>35\text{ dB}$). |
        """)

    with tab_v2:
        st.markdown("""
        ### 🎯 Live Viva Demonstration Workflow (5-Minute Walkthrough)

        1. **Step 1: Acquisition (Page 2)**
           - Select **Vocal / Speech Formant** and inject **50 Hz Powerline Hum** at `10 dB` SNR.
           - Play the noisy audio so the evaluator clearly hears the loud low-frequency hum.
        2. **Step 2: Spectral & Z-Transform Analysis (Pages 3 & 4)**
           - Show the **FFT spectrum** with prominent harmonic spikes at 50 Hz, 100 Hz, 150 Hz.
           - Navigate to **Z-Transform Modeling** to show the Laurent polynomial expansion and the complex zeros located in the Z-plane.
        3. **Step 3: Filter Design in the Z-Domain (Page 5)**
           - Select **Butterworth IIR High-pass** (or **Digital Notch Filter**) with cutoff at `80 Hz` (Order 4).
           - Point out the **Pole-Zero Diagram**: all poles lie strictly inside the unit circle ($|p_k| < 1.0$), proving **BIBO Stability**.
           - Show the transfer function $H(z)$ and difference equation $y[n]$.
        4. **Step 4: Audio Enhancement & Playback (Page 6)**
           - Switch to **filtfilt (Zero-Phase)** filtering.
           - Play the **Enhanced Audio $y[n]$**: the hum is completely gone!
           - Play the **Residual $e[n]$**: prove that only the isolated 50 Hz hum was removed.
           - Show the **Dual Spectrograms** where the 50 Hz horizontal line vanishes.
        5. **Step 5: Quantitative Evaluation (Page 7)**
           - Demonstrate the exact **$\Delta\text{SNR}$ improvement** (e.g. $+9.2\text{ dB}$) and **Stopband Attenuation** ($>40\text{ dB}$).
        """)

    with tab_v3:
        st.markdown("""
        ### 💡 Evaluator Questions & Mathematical Answers

        **Q1: Why use the Z-Transform instead of just applying FFT?**
        > *Answer:* FFT only evaluates frequencies along the unit circle ($z = e^{j\omega}$). The Z-transform generalizes this to the entire complex plane ($z = r e^{j\omega}$), enabling us to analyze system stability via pole locations, compute Regions of Convergence (ROC), and mathematically synthesize difference equations $y[n] = \sum b_k x[n-k] - \sum a_k y[n-k]$.

        **Q2: What guarantees that an IIR filter will not blow up or oscillate?**
        > *Answer:* The Bounded-Input Bounded-Output (BIBO) Stability Theorem requires that all poles of the transfer function $H(z) = B(z)/A(z)$ lie strictly inside the unit circle: $\max_k |p_k| < 1.000$. If any pole reaches $|p_k| \ge 1$, the impulse response $h[n]$ does not decay to zero as $n \to \infty$.

        **Q3: What is the difference between `lfilter` and `filtfilt`?**
        > *Answer:* `lfilter` applies causal single-pass filtering simulating real-time DSP, which introduces frequency-dependent phase delay. `filtfilt` filters the signal forward and then backward in time, which cancels out all phase distortion ($\angle H_{eff}(e^{j\omega}) = 0$) and doubles the effective filter order roll-off.
        """)
