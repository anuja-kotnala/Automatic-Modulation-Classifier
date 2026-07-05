# Automatic Modulation Classification (AMC) & Spectrum Analyzer

A research-grade, configuration-driven, object-oriented framework for simulating baseband signals, applying wireless channel impairments, extracting statistical and spectral features, performing modulation classification (ML/DL), and visualizing spectrum dynamics.

## Supported Modulation Schemes
- **Analog:** Amplitude Modulation (AM), Frequency Modulation (FM)
- **Digital Single-Carrier:** BPSK, QPSK, 16QAM, 64QAM
- **Multi-Carrier:** OFDM (Orthogonal Frequency Division Multiplexing)

## Folder & Package Structure

```
d:/ODIN/AMC/
├── configs/
│   └── default_config.yaml         # Project configuration (parameters, thresholds)
├── dataset/                        # Generated dataset (raw signal files and metadata indexes)
│   ├── raw/
│   ├── metadata.csv
│   └── features.csv
├── results/                        # Generated evaluation outputs and model weights
│   ├── analysis/                   # Dimensionality reduction and feature quality rankings
│   ├── plots/                      # Publication-quality signal diagrams
│   ├── ml_eval/                    # Machine learning evaluation figures and performance summary
│   ├── dl_eval/                    # Deep learning training learning curves and confusion matrices
│   └── models/                     # Serialized model binaries (ML / DL PyTorch checkpoints)
├── amc/                            # Main python package
│   ├── __init__.py                 # Exports package parameters, classes
│   ├── config.py                   # Type-safe configurations (dataclasses)
│   ├── constants.py                # Modulation types and mathematical constants
│   ├── logging_config.py           # Logging initialization
│   ├── core/                       # Base abstract classes specifying standard interfaces
│   │   ├── __init__.py
│   │   ├── base_generator.py       # BaseModulator interface
│   │   ├── base_extractor.py       # BaseFeatureExtractor interface
│   │   ├── base_classifier.py      # BaseClassifier interface
│   │   └── base_analyzer.py        # BaseSpectrumAnalyzer interface
│   ├── generator/                  # Waveform modulators (AM, FM, BPSK, QPSK, QAM, OFDM)
│   │   ├── __init__.py
│   │   ├── am.py
│   │   ├── fm.py
│   │   ├── bpsk.py
│   │   ├── qpsk.py
│   │   ├── qam16.py
│   │   ├── qam64.py
│   │   └── ofdm.py
│   ├── channel/                    # Channel impairments (AWGN, fading, CFO, IQ mismatch, drift)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── awgn.py
│   │   ├── rayleigh.py
│   │   ├── rician.py
│   │   ├── frequency_offset.py
│   │   ├── phase_noise.py
│   │   ├── iq_imbalance.py
│   │   ├── timing_offset.py
│   │   ├── multipath.py
│   │   ├── clock_drift.py
│   │   └── channel_pipeline.py
│   ├── pulse_shaping/              # Pulse shaping filters (Raised Cosine, RRC, Rectangular)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── raised_cosine.py
│   │   ├── root_raised_cosine.py
│   │   ├── rectangular.py
│   │   └── filter_utils.py
│   ├── features/                   # Feature extractors (Cumulants, Wavelet, Spectral)
│   │   ├── __init__.py
│   │   ├── extractors.py
│   │   └── feature_definitions.md
│   ├── classifier/                 # Core classifier wrappers
│   │   ├── __init__.py
│   │   └── models.py
│   ├── analyzer/                   # Spectrum Analyzer implementations (Welch PSD, Spectrograms, Bandwidth)
│   │   ├── __init__.py
│   │   └── spectrum.py
│   └── utils/                      # Helper modules
│       ├── __init__.py
│       ├── dsp_utils.py            # AWGN noise addition, filter design, normalization
│       ├── io.py                   # SigMF-like binary and json loading/saving
│       └── visualization.py        # Constellations, PSD plots, confusion matrices
├── tests/                          # Test suite matching structure
├── requirements.txt                # Third-party dependencies
├── main_generate_dataset.py        # Dataset generation pipeline script
├── extract_features.py             # Feature compiler script
├── analyze_features.py             # Feature analysis and ranking script
├── train_ml.py                     # Machine learning pipeline training script
├── train_dl.py                     # PyTorch deep learning pipeline script
├── generate_plots.py               # Signal diagram generator script
└── README.md                       # Documentation
```

## Setup and Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate realistic SDR Dataset:**
   ```bash
   python main_generate_dataset.py
   ```
   Generates 6,300 complex signal files under `dataset/raw/` and records parameters inside `dataset/metadata.csv`.

3. **Extract Features:**
   ```bash
   python extract_features.py
   ```
   Computes 39 statistical, spectral, phase, and HOC cumulant features in parallel and saves `dataset/features.csv`.

4. **Analyze and Rank Features:**
   ```bash
   python analyze_features.py
   ```
   Generates PCA/t-SNE/UMAP dimension reductions, feature vs SNR plots, and ranks feature classifications inside `results/analysis/`.

5. **Train Machine Learning Classifiers:**
   ```bash
   python train_ml.py
   ```
   Trains and evaluates Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM, KNN, and Logistic Regression, saving performance reports under `results/ml_eval/`.

6. **Train Deep Learning Models:**
   - On Raw IQ signals:
     ```bash
     python train_dl.py --input_type raw --model cnn1d
     ```
   - On Extracted features:
     ```bash
     python train_dl.py --input_type features --model cnn1d
     ```

7. **Generate Diagnostic Signal Diagrams:**
   ```bash
   python generate_plots.py
   ```
   Renders and saves constellation scatter, PSD, STFT spectrogram, and 3D waterfall plots under `results/plots/`.

8. **Execute Validation Test Suite:**
   ```bash
   python -m unittest discover -s tests
   ```

## Design Philosophy

- **OOP Design & Modularity:** Clean interfaces (`amc/core/`) separate waveform generation, feature calculation, channel models, and classification.
- **Configuration-Driven:** Strict, typed Python `dataclasses` sync settings from `configs/default_config.yaml`.
- **Publication-Quality Visualizations:** Matplotlib diagrams feature clean scientific styling, custom grids, and balanced scales.
