# Audio Signal Enhancement Using Z-Transform Based Digital Filtering
### Applied Mathematics & Digital Signal Processing Capstone Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)
![NumPy](https://img.shields.io/badge/NumPy-DSP-013243.svg)
![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-0054A6.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)

---

## 1. Project Overview & Problem Statement

Audio signals acquired in real-world environments are frequently corrupted by acoustic noise, electrical powerline interference (50 Hz / 60 Hz hum), high-frequency electronic whine, and broadband white Gaussian noise. 

This capstone project implements a **mathematically rigorous, fully functional digital signal processing application** that demonstrates how **Z-transform theory**, **Laurent series representations**, **pole-zero placement**, and **digital filter design (FIR and IIR)** are applied to discrete-time audio signals to isolate and eliminate noise while preserving signal fidelity.

Unlike generic audio editors or mock dashboards, this system exposes the complete underlying discrete mathematics at every stage:
- Computing the finite Laurent series polynomial $X(z) = \sum_{n=0}^{N-1} x[n] z^{-n}$ from real audio samples.
- Locating complex zeros of the discrete audio sequence in the $z$-plane.
- Evaluating the 2D complex manifold $|X(z)|$ and proving its cross-sectional equivalence to the Discrete-Time Fourier Transform (DTFT) along the unit circle $|z|=1$.
- Synthesizing stable digital transfer functions $H(z) = \frac{B(z)}{A(z)}$ with verified BIBO stability ($\max_k |p_k| < 1.0$).
- Applying linear constant-coefficient difference equations (LCCDE) in causal (`lfilter`) and zero-phase (`filtfilt`) implementations.
- Quantifying performance using exact $\Delta\text{SNR}$, Mean Squared Error (MSE), and stopband rejection (dB).

---

## 2. System Architecture & Workflow

```
+-------------------------------------------------------------+
|                      1. AUDIO ACQUISITION                   |
|  - File Upload (WAV, MP3, OGG, FLAC)                        |
|  - Academic Synthetic Generator (Speech Formant, Harmonics) |
|  - Calibrated Noise Injection (50Hz Hum, Whine, AWGN)       |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    2. PREPROCESSING x[n]                    |
|  - Stereo-to-Mono Averaging: x[n] = 0.5 * (x_L[n] + x_R[n]) |
|  - Peak Normalization to [-1.0, 1.0]                        |
|  - Discrete Time Representation: x[n] = x_a(n * T_s)        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               3. TIME & SPECTRAL DOMAIN ANALYSIS            |
|  - Statistical Moments (RMS, Peak, Crest Factor, Energy)    |
|  - N-point FFT: X[k] = \sum x[n] w[n] e^{-j 2\pi k n / N}   |
|  - Dominant Harmonic Peak Identification & Centroid         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               4. Z-TRANSFORM MATHEMATICAL MODELING          |
|  - Laurent Polynomial Expansion: X(z) = \sum x[n] z^{-n}    |
|  - Complex Zero Extraction: Roots of P(z) = 0               |
|  - Complex Z-Plane Manifold Evaluation & Unit Circle DTFT   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|            5. DIGITAL FILTER DESIGN IN THE Z-DOMAIN         |
|  - Rational Transfer Function: H(z) = B(z) / A(z)           |
|  - Filter Types: Butterworth, Chebyshev I & II, FIR, Notch  |
|  - Pole-Zero Placement & Unit Circle Constellation          |
|  - BIBO Stability Verification: \max |p_k| < 1.000          |
|  - Magnitude |H(e^{j\omega})| & Phase \angle H(e^{j\omega}) |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              6. FILTER EXECUTION & AUDIO ENHANCEMENT        |
|  - LCCDE Difference Equation Execution                      |
|  - Causal (lfilter) vs Zero-Phase (filtfilt) Processing     |
|  - In-browser Audio Playback & 16-bit PCM WAV Export        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             7. BEFORE/AFTER & PERFORMANCE EVALUATION        |
|  - Stacked Waveforms: x[n] vs y[n] vs e[n] (Residual)       |
|  - Dual Spectral Overlay & Synchronized Dual Spectrograms   |
|  - True \Delta SNR (dB), MSE Reduction (%), Stopband Atten. |
+-------------------------------------------------------------+
```

---

## 3. Mathematical Foundations

### 3.1 Discrete-Time Representation
The continuous audio waveform $x_a(t)$ is sampled at uniform intervals $T_s = 1/f_s$:
$$x[n] = x_a(n T_s), \quad n \in \{0, 1, 2, \dots, N-1\}$$

### 3.2 Discrete Fourier Transform (DFT / FFT)
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot w[n] \cdot e^{-j \frac{2\pi k n}{N}}, \quad k \in \{0, 1, \dots, N-1\}$$
where frequency corresponding to bin $k$ is $f_k = \frac{k \cdot f_s}{N}$.

### 3.3 Z-Transform Laurent Polynomial
The bilateral Z-transform of a finite sequence $x[n]$ of length $N$ is given by:
$$X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=0}^{N-1} x[n] z^{-n} = x[0] + x[1]z^{-1} + x[2]z^{-2} + \dots + x[N-1]z^{-(N-1)}$$
Expressed as a rational polynomial in $z$:
$$X(z) = \frac{x[0]z^{N-1} + x[1]z^{N-2} + \dots + x[N-1]}{z^{N-1}}$$
The Region of Convergence (ROC) is the entire complex plane except $z=0$ ($\text{ROC}: |z| > 0$).

### 3.4 Transfer Function in the Z-Domain & Difference Equation
A Linear Time-Invariant (LTI) digital filter is governed by the Linear Constant-Coefficient Difference Equation (LCCDE):
$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$
Applying the Z-transform linearity and time-shifting properties $\mathcal{Z}\{x[n-k]\} = z^{-k} X(z)$:
$$Y(z) = \sum_{k=0}^{M} b_k z^{-k} X(z) - \sum_{k=1}^{N} a_k z^{-k} Y(z)$$
$$H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}} = K \frac{\prod_{i=1}^{M} (1 - q_i z^{-1})}{\prod_{k=1}^{N} (1 - p_k z^{-1})}$$
where $q_i$ are system zeros and $p_k$ are system poles.

### 3.5 BIBO Stability Criterion
A causal digital filter is Bounded-Input Bounded-Output (BIBO) stable if and only if all poles of $H(z)$ lie strictly within the unit circle:
$$\max_{k} |p_k| < 1.0000$$

### 3.6 Quantitative Metrics
1. **Root Mean Square (RMS)**:
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2[n]}$$
2. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{N} \sum_{n=0}^{N-1} (y[n] - x_{clean}[n])^2$$
3. **Signal-to-Noise Ratio (SNR) Improvement**:
   $$\text{SNR}_{\text{in, dB}} = 10 \log_{10}\left( \frac{\sum x_{clean}^2[n]}{\sum (x_{noisy}[n] - x_{clean}[n])^2} \right)$$
   $$\text{SNR}_{\text{out, dB}} = 10 \log_{10}\left( \frac{\sum x_{clean}^2[n]}{\sum (y[n] - x_{clean}[n])^2} \right)$$
   $$\Delta \text{SNR} = \text{SNR}_{\text{out, dB}} - \text{SNR}_{\text{in, dB}}$$

---

## 4. 4-Module Structure Mapping to Capstone PPT

| Module | Title | Key Components |
| :--- | :--- | :--- |
| **Module 1** | **Audio Analysis & Z-Transform Modeling** | Audio acquisition, normalization, discrete-time representation $x[n]$, FFT spectral decomposition, finite sequence Laurent polynomial $X(z)$, polynomial zeros extraction, 2D Z-plane manifold & unit circle DTFT equivalence. |
| **Module 2** | **Digital Filter Design (Z-Domain)** | Synthesis of FIR (Hamming, Hann, Blackman) & IIR (Butterworth, Chebyshev I & II, Elliptic, Notch) filters, transfer function $H(z)$, pole-zero constellation, BIBO stability verification ($\max \|p_k\| < 1.0$). |
| **Module 3** | **Digital Audio Filtering Implementation** | Difference equation execution, Direct Form II causal filtering (`lfilter`), zero-phase forward-backward filtering (`filtfilt`), residual noise separation $e[n] = x[n] - y[n]$, in-memory 16-bit PCM WAV encoding. |
| **Module 4** | **Performance Evaluation & Quality Analysis** | Stacked waveform comparisons, spectral overlays, dual time-frequency spectrograms, exact $\Delta\text{SNR}$ improvement (dB), stopband rejection (dB), MSE reduction (%). |

---

## 5. Project Directory Structure

```
applied/
│
├── app.py                      # Streamlit Scientific Dashboard Application
├── requirements.txt            # Python dependencies
├── README.md                   # Academic Documentation & Viva Guide
├── test_pipeline.py            # Automated mathematical & DSP verification test suite
│
└── modules/
    ├── __init__.py             # Package initializer
    ├── audio_processing.py     # Audio I/O, normalization, synthesis, calibrated noise injection
    ├── fft_analysis.py         # Discrete Fourier Transform, peak detection, spectral moments
    ├── z_transform.py          # Laurent polynomial expansion, zero-finding, complex Z-plane manifold
    ├── filter_design.py        # FIR & IIR filter synthesis, pole-zero extraction, stability validation
    ├── filter_execution.py     # Causal lfilter & Zero-Phase filtfilt execution, difference equations
    ├── performance.py          # SNR (exact & proxy), MSE, RMS, Peak, stopband attenuation metrics
    └── visualization.py        # Publication-grade dark-navy & cyan DSP plotting library
```

---

## 6. Installation & Execution

### Prerequisites
- Python 3.9 or higher installed.

### Step 1: Clone or Navigate to Directory
```bash
cd c:/projects/applied
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
```bash
python test_pipeline.py
```

### Step 4: Launch the Streamlit Application
```bash
streamlit run app.py
```

---

## 7. Viva Demonstration Workflow (5 Minutes)

When presenting before the examination committee, follow this exact workflow:

1. **Step 1: Audio Acquisition & Noise Generation (Page 2)**
   - Select **Vocal / Speech Formant** and inject **50 Hz Powerline Hum** at `10 dB` SNR.
   - Click **Generate & Load Synthetic Audio**.
   - Play the noisy audio so the evaluators hear the low-frequency hum.

2. **Step 2: Time & Spectral Analysis (Page 3)**
   - Highlight the **FFT Magnitude Spectrum** with distinct harmonic spikes at $50\text{ Hz}, 100\text{ Hz}, 150\text{ Hz}$.

3. **Step 3: Z-Transform Modeling (Page 4)**
   - Show the Laurent polynomial:
     $$X(z) = x[0] + x[1]z^{-1} + x[2]z^{-2} + \dots$$
   - Point out the **Complex Z-Plane Zeros** and the **2D Z-Plane Manifold** proving that the unit circle $|z|=1$ corresponds directly to the DTFT.

4. **Step 4: Digital Filter Design (Page 5)**
   - Select **Butterworth (IIR)** -> **High-pass** -> Cutoff at `80 Hz` (Order 4) or **Digital Notch Filter** at `50 Hz`.
   - Show the **Pole-Zero Plot**: all poles are strictly inside the unit circle ($|p_k| < 1.0$), confirming **BIBO Stability**.
   - Show the resulting $H(z)$ transfer function and difference equation $y[n]$.

5. **Step 5: Filtering & Playback (Page 6)**
   - Select **filtfilt (Zero-Phase)** algorithm.
   - Play the **Enhanced Audio $y[n]$**: the hum is completely removed while the vocal tone remains crisp.
   - Play the **Residual $e[n]$**: demonstrate that only the isolated 50 Hz hum was extracted.
   - Show the **Dual Spectrograms** where the 50 Hz horizontal line disappears.

6. **Step 6: Quantitative Performance Evaluation (Page 7)**
   - Show the **$\Delta\text{SNR}$ improvement** ($+9.2\text{ dB}$ to $+12\text{ dB}$), **MSE reduction** ($>85\%$), and **Stopband Attenuation** ($>40\text{ dB}$).

---

## 8. Key Viva Questions & Mathematical Answers

- **Q: Why use the Z-transform instead of just the Fourier transform?**
  - *A:* The Fourier transform evaluates signals only on the boundary of the unit circle ($|z|=1$). The Z-transform generalizes to the entire complex plane ($z = r e^{j\omega}$), which is necessary to analyze filter stability via pole locations, compute transient dynamics, determine Regions of Convergence (ROC), and implement difference equations.

- **Q: What determines whether an IIR filter is BIBO stable?**
  - *A:* A causal filter is BIBO stable if and only if its Region of Convergence includes the unit circle, which requires all poles $\{p_k\}$ of $H(z) = \frac{B(z)}{A(z)}$ to satisfy $|p_k| < 1.000$.

- **Q: What is the mathematical advantage of `filtfilt` over `lfilter`?**
  - *A:* `lfilter` performs single-pass causal filtering: $Y(z) = H(z)X(z)$, introducing nonlinear phase delay. `filtfilt` performs forward and time-reversed backward filtering: $Y(z) = H(z) H(z^{-1}) X(z)$, yielding an effective transfer function $|H(e^{j\omega})|^2$ with strictly zero phase distortion ($\angle H_{\text{eff}} = 0$).

---

## 9. Limitations & Future Scope

### Current Limitations
- Filter synthesis is designed for Linear Time-Invariant (LTI) stationary noise profiles.
- Very high IIR filter orders ($N > 12$) in Direct Form II may experience coefficient quantization sensitivity (second-order sections SOS format can be extended for higher orders).

### Future Scope
- Adaptive digital filtering (LMS / RLS algorithms) for non-stationary acoustic environments.
- Wavelet Transform (DWT / CWT) hybrid Z-domain multi-resolution thresholding.
- Hardware DSP deployment on embedded ARM / FPGA microcontrollers.
