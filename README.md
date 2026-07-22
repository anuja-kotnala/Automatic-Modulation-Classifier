# Automatic Modulation Classification (AMC) & Spectrum Analyzer

An end-to-end, research-grade, object-oriented framework for simulating complex baseband signals, applying realistic wireless channel impairments, extracting statistical & spectral features, training machine learning/deep learning modulation classifiers, and visualizing spectrum dynamics in an interactive dashboard.

## 📺 Demonstration Video

Here is a short demonstration walkthrough showing the application features, real-time AMC predictions, model training phases, and evaluation results.

<video src="docs/demo.mp4" controls width="100%"></video>

---

## 🌟 Key Features

1. **Baseband Signal Simulation:** High-fidelity waveform generators supporting:
   * **Analog Modulation:** AM, FM
   * **Digital Single-Carrier:** BPSK, QPSK, 16QAM, 64QAM
   * **Multi-Carrier:** OFDM (Orthogonal Frequency Division Multiplexing)
2. **Impairing Channel Emulator:** Realistic channel impairments including:
   * Additive White Gaussian Noise (AWGN)
   * Fading Channels (Rayleigh, Rician multipath propagation)
   * Carrier Frequency Offset (CFO) & Phase Noise
   * IQ Amplitude & Phase Imbalance
   * Timing offsets, multipath delays, and clock drifts
3. **Statistical & Spectral Feature Extraction:** Automatic compilation of 39 distinct statistical features (moments, cumulants, envelope features, wavelet energy levels, and PSD statistics) for classical classification.
4. **Machine Learning Classifiers:** Classical models (Random Forest, SVM, LightGBM, XGBoost, etc.) trained, cross-validated, and serialized for real-time inference.
5. **Deep Learning Architectures:** Raw IQ classification models using PyTorch, including 1D CNNs, CNN-LSTMs, and 2D CNN (spectrogram) architectures.
6. **Unified Web Dashboard:** Modern dashboard (Vite + React + Material-UI) showing:
   * Pipeline steps (Dataset Generation ➔ Feature Extraction ➔ Ranking ➔ ML/DL Classification)
   * Live AMC Predictor (interactive signal generation, impairment addition, and instant ML/DL classification)
   * Model Performance analytics & confusion matrices

---

## 📁 Repository Structure

```text
d:/ODIN/AMC/
├── docs/
│   └── demo.mp4                    # Walkthrough demonstration video
├── amc-backend/                    # FastAPI REST backend service
│   ├── amc/                        # Core Python library
│   │   ├── generator/              # Waveform modulators
│   │   ├── channel/                # Channel impairment pipeline
│   │   ├── features/               # Envelope/spectral/HOC extractors
│   │   ├── pulse_shaping/          # RC/RRC filtering
│   │   ├── classifier/             # ML model wrappers
│   │   └── analyzer/               # Spectrum analysis (Welch PSD, Spectrograms)
│   ├── api/                        # API route and inference controllers
│   ├── configs/                    # Default configuration YAMLs
│   ├── dataset/                    # Generated datasets (metadata.csv, features.csv)
│   ├── results/                    # Compiled models, training plots, and CSV summaries
│   ├── app.py                      # REST API server script
│   └── requirements.txt            # Python dependencies
└── amc-frontend/                   # Vite + React + Material-UI web interface
    ├── src/
    │   ├── api/                    # Axios-based backend integration clients
    │   ├── pages/                  # Dashboard, Predictor, Results, Performance pages
    │   └── components/             # Reusable UI layout elements
    ├── package.json                # Node package dependencies
    └── vite.config.ts              # Vite configuration
```

---

## 🚀 Setup & Execution

### Deployed frontend: https://automatic-modulation-classifier.vercel.app/

### Deployed backend: https://amc-backend-ilfa.onrender.com/

### 1. Backend API (FastAPI)

Ensure you have Python 3.10+ installed.

1. Navigate to the backend directory:
   ```bash
   cd amc-backend
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dataset generator, feature extractor, and train the classifiers (if not already trained):
   ```bash
   python main_generate_dataset.py
   python extract_features.py
   python train_ml.py
   python train_dl.py
   ```
4. Start the REST API server:
   ```bash
   python app.py
   ```
   *The server runs locally by default on `http://localhost:8000`.*

### 2. Frontend Interface (Vite + React)

Ensure you have Node.js (v18+) installed.

1. Navigate to the frontend directory:
   ```bash
   cd amc-frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *Access the web app at `http://localhost:5173`.*

---

## 📊 Evaluation & Metrics

The system saves comprehensive figures and summaries under `amc-backend/results/`:
- **Feature Importance:** Heatmaps showing Pearson/Spearman feature correlations.
- **ML Performance:** Accuracy, Precision, Recall, F1 scores, Cross-validation scores, and Confusion Matrices.
- **DL Performance:** Loss/accuracy curves and SNR-specific performance profiles.
