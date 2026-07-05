# Feature Definitions

This document details the mathematical and conceptual definition of each of the 34 features extracted by the AMC Feature Extraction Framework.

## 1. Time-Domain Statistical Features (Real/Imaginary)
*   **`mean_i` / `mean_q`**: The average value of the In-phase (Real) and Quadrature (Imaginary) components of the signal. Useful to identify DC offsets.
*   **`var_i` / `var_q`**: The variance (spread) of the I and Q components.
*   **`std_i` / `std_q`**: The standard deviation of the I and Q components.
*   **`skew_i` / `skew_q`**: The skewness (asymmetry) of the I and Q distributions around their means.
*   **`kurt_i` / `kurt_q`**: The kurtosis (tailedness/heaviness of outliers) of the I and Q components.

## 2. Envelope & Amplitude Features
*   **`env_mean`**: The mean of the signal envelope (amplitude magnitude $A = \sqrt{I^2 + Q^2}$).
*   **`env_var`**: The variance of the envelope.
*   **`env_std`**: The standard deviation of the envelope.
*   **`rms`**: The Root Mean Square value of the complex signal.
*   **`papr`**: Peak-to-Average Power Ratio in dB, defined as $10 \log_{10} (\max(|x|^2) / \text{mean}(|x|^2))$. Crucial for differentiating OFDM from single-carrier systems.
*   **`energy`**: Total energy of the signal, calculated as the sum of squared magnitudes $\sum |x_n|^2$.
*   **`power`**: Average power of the signal, calculated as $\frac{1}{N} \sum |x_n|^2$.

## 3. Instantaneous Phase & Frequency Features
*   **`inst_phase_mean`**: Average of the instantaneous phase $\theta_n = \text{arg}(I_n + j Q_n)$.
*   **`inst_phase_std`**: Standard deviation of the instantaneous phase. Highly effective for separating BPSK, QPSK, and QAM.
*   **`inst_freq_mean`**: Average of the instantaneous frequency, computed as the derivative of phase $f_n = \frac{\Delta \theta_n}{2\pi}$.
*   **`inst_freq_std`**: Standard deviation of the instantaneous frequency. Indicates frequency variations (e.g. FM).

## 4. Crossing & Physical Features
*   **`zero_crossing_rate`**: The rate at which the real part of the signal changes sign. Useful for frequency and bandwidth estimation.

## 5. Spectral Features
*   **`spectral_entropy`**: The Shannon entropy of the normalized Power Spectral Density. Quantifies the randomness/noise-like quality of the spectrum.
*   **`spectral_flatness`**: The ratio of the geometric mean of the PSD to the arithmetic mean. Indicates if the spectrum is tonal (low flatness) or noise-like (high flatness close to 1).
*   **`spectral_centroid`**: The "center of mass" of the spectrum, indicating where the bulk of the signal power is centered.
*   **`bandwidth_3db`**: The 3dB (half-power) bandwidth calculated from the PSD.
*   **`occupied_bandwidth_99`**: The occupied bandwidth containing 99% of the total integrated power.
*   **`peak_frequency`**: The frequency index where the PSD reaches its maximum power level.

## 6. High-Order Statistics & Cumulants
High-order cumulants (HOC) are extremely powerful features for digital modulation classification.
*   **`C20`**: Second-order cumulant, defined as $E[x^2]$.
*   **`C21`**: Second-order cumulant, defined as $E[|x|^2]$ (average power).
*   **`C40`**: Fourth-order cumulant, defined as $E[x^4] - 3(E[x^2])^2$.
*   **`C41`**: Fourth-order cumulant, defined as $E[x^3 x^*] - 3E[x^2]E[|x|^2]$.
*   **`C42`**: Fourth-order cumulant, defined as $E[|x|^4] - |E[x^2]|^2 - 2(E[|x|^2])^2$.
*   **`C60`**: Sixth-order cumulant, defined as $E[x^6] - 15 E[x^2] E[x^4] + 30 (E[x^2])^3$.
*   **`C61`**: Sixth-order cumulant, defined as $E[x^5 x^*] - 5 E[x^2] E[x^3 x^*] - 10 E[|x|^2] E[x^4] + 30 E[|x|^2] (E[x^2])^2$.
*   **`C62`**: Sixth-order cumulant, defined as $E[x^4 (x^*)^2] - 4 E[x^2] E[x^2 (x^*)^2] - E[x^2] E[x^4] - 8 E[|x|^2] E[x^3 x^*] + 24 (E[x^2])^2 E[|x|^2]$.
*   **`C63`**: Sixth-order cumulant, defined as $E[|x|^6] - 9 E[|x|^2] E[|x|^4] - 6 E[|x|^2] |E[x^2]|^2 + 12 (E[|x|^2])^3$.

## 7. Cyclostationary Features
*   **`cyclic_max_alpha`**: The maximum peak value of the Cyclic Autocorrelation Function (CAF) across non-zero cyclic frequencies $\alpha$. Identifies periodic keying properties.
*   **`cyclic_mean`**: The mean intensity of cyclic correlation.
