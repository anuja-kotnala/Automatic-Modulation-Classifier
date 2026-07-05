import React, { useState, useEffect } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  MenuItem,
  Slider,
  FormControlLabel,
  Switch,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  LinearProgress,
  Divider,
  ListSubheader,
} from "@mui/material";
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Analytics as AnalyticsIcon,
  Image as ImageIcon,
  ListAlt as ListIcon,
  Timer as TimerIcon,
} from "@mui/icons-material";
import { amcApi } from "../services/amcApi";
import type {
  ChannelImpairments,
  PredictResponse,
  GeneratedPlotItem,
} from "../services/amcApi";
import { API_BASE_URL } from "../api/client";
import AcademicHeader from "../components/Common/AcademicHeader";
import LoadingSkeleton from "../components/Common/LoadingSkeleton";
import EmptyState from "../components/Common/EmptyState";

export const AmcPredictor: React.FC = () => {
  // Main Signal params
  const [modulation, setModulation] = useState("BPSK");
  const [snr, setSnr] = useState(20);
  const [sampleRate, setSampleRate] = useState(1000000);
  const [numSamples, setNumSamples] = useState(1024);
  const [selectedModel, setSelectedModel] = useState("randomforest");
  const [availableMLModels, setAvailableMLModels] = useState<string[]>([]);
  const [availableDLModels, setAvailableDLModels] = useState<string[]>([]);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const mlHealth = await amcApi.getHealth();
        const mlModels = Object.keys(mlHealth.models_loaded).filter(
          (k) => mlHealth.models_loaded[k],
        );
        setAvailableMLModels(mlModels);
      } catch (err) {
        console.warn("Failed to load health status of ML models", err);
        setAvailableMLModels([
          "randomforest",
          "svm",
          "gradientboosting",
          "xgboost",
          "lightgbm",
          "knn",
          "logisticregression",
        ]);
      }

      try {
        const dlHealth = await amcApi.getHealthDL();
        const dlModels = Object.keys(dlHealth.models_loaded).filter(
          (k) => dlHealth.models_loaded[k],
        );
        setAvailableDLModels(dlModels);
      } catch (err) {
        console.warn("Failed to load health status of DL models", err);
        setAvailableDLModels(["cnn1d", "cnnlstm", "cnn2d"]);
      }
    };
    fetchHealth();
  }, []);

  // Impairments switches and values
  const [awgnEnabled, setAwgnEnabled] = useState(true);
  const [rayleighEnabled, setRayleighEnabled] = useState(false);
  const [rayleighDoppler, setRayleighDoppler] = useState(10.0);
  const [ricianEnabled, setRicianEnabled] = useState(false);
  const [ricianK, setRicianK] = useState(4.0);
  const [ricianDoppler, setRicianDoppler] = useState(5.0);
  const [freqOffsetEnabled, setFreqOffsetEnabled] = useState(false);
  const [freqOffset, setFreqOffset] = useState(50.0);
  const [phaseNoiseEnabled, setPhaseNoiseEnabled] = useState(false);
  const [phaseNoiseStd, setPhaseNoiseStd] = useState(0.05);
  const [iqImbalanceEnabled, setIqImbalanceEnabled] = useState(false);
  const [iqAmpImb, setIqAmpImb] = useState(0.5);
  const [iqPhaseImb, setIqPhaseImb] = useState(3.0);
  const [timingOffsetEnabled, setTimingOffsetEnabled] = useState(false);
  const [timingOffset, setTimingOffset] = useState(0.25);
  const [multipathEnabled, setMultipathEnabled] = useState(false);
  const [multipathDelays, setMultipathDelays] = useState("0.0, 1.2e-6, 2.5e-6");
  const [multipathGains, setMultipathGains] = useState("0.0, -3.0, -9.0");
  const [clockDriftEnabled, setClockDriftEnabled] = useState(false);
  const [clockDrift, setClockDrift] = useState(20.0);

  // States
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const [plots, setPlots] = useState<GeneratedPlotItem[]>([]);
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleSimulateAndPredict = async () => {
    setLoading(true);
    setError(null);
    const startTime = performance.now();
    try {
      // 1. Build channel impairments
      const impairments: ChannelImpairments = {
        awgn_enabled: awgnEnabled,
        awgn_snr_db: snr,
        rayleigh_enabled: rayleighEnabled,
        rayleigh_doppler_freq: rayleighDoppler,
        rician_enabled: ricianEnabled,
        rician_k_factor: ricianK,
        rician_doppler_freq: ricianDoppler,
        frequency_offset_enabled: freqOffsetEnabled,
        frequency_offset_hz: freqOffset,
        phase_noise_enabled: phaseNoiseEnabled,
        phase_noise_std_dev: phaseNoiseStd,
        iq_imbalance_enabled: iqImbalanceEnabled,
        iq_imbalance_amplitude_db: iqAmpImb,
        iq_imbalance_phase_deg: iqPhaseImb,
        timing_offset_enabled: timingOffsetEnabled,
        timing_offset_fractional_delay: timingOffset,
        multipath_enabled: multipathEnabled,
        clock_drift_enabled: clockDriftEnabled,
        clock_drift_ppm: clockDrift,
      };

      if (multipathEnabled) {
        impairments.multipath_delays = multipathDelays
          .split(",")
          .map((x) => parseFloat(x.trim()));
        impairments.multipath_gains_db = multipathGains
          .split(",")
          .map((x) => parseFloat(x.trim()));
      }

      // 2. Generate signal
      const signalRes = await amcApi.generateSignal({
        modulation_type: modulation,
        snr_db: snr,
        sample_rate: sampleRate,
        num_samples: numSamples,
        impairments,
      });

      if (!signalRes.success) {
        throw new Error("Signal generation failed on the backend.");
      }

      // 3. Classify signal using generated IQ data
      const isDL = ["cnn1d", "cnnlstm", "cnn2d"].includes(selectedModel);
      const predictRes = isDL
        ? await amcApi.predictDL({
            iq_data: signalRes.iq_data,
            model_name: selectedModel,
          })
        : await amcApi.predict({
            iq_data: signalRes.iq_data,
            model_name: selectedModel,
          });

      setPrediction(predictRes);

      // Measure elapsed time for predictor inference latency in ms
      const endTime = performance.now();
      setInferenceTime(Math.round(endTime - startTime));

      // 4. Generate diagnostic plots
      const plotRes = await amcApi.generatePlots({
        iq_data: signalRes.iq_data,
        modulation_type: modulation,
        sample_rate: sampleRate,
      });

      if (plotRes.success) {
        setPlots(plotRes.plots);
      }
    } catch (err: any) {
      setError(
        err?.message ||
          "Error occurred during classification simulation. Please check backend log.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AcademicHeader
        title="AMC Predictor & Channel Simulator"
        subtitle="Simulate impaired signals on the fly, extract features, and run real-time machine learning predictions."
      />

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={4}>
        {/* Left Side: Parameters & Impairments */}
        <Grid
          size={{ xs: 12, lg: 4 }}
          sx={{ display: "flex", flexDirection: "column", gap: 3 }}
        >
          <Card>
            <CardContent
              sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                1. Transmitter Parameters
              </Typography>
              <Divider />

              <TextField
                select
                label="Modulation Type"
                value={modulation}
                onChange={(e) => setModulation(e.target.value)}
                fullWidth
              >
                {[
                  "BPSK",
                  "QPSK",
                  "8PSK",
                  "QAM16",
                  "QAM64",
                  "CPFSK",
                  "GFSK",
                  "PAM4",
                  "AM-DSB",
                  "FM",
                ].map((item) => (
                  <MenuItem key={item} value={item}>
                    {item}
                  </MenuItem>
                ))}
              </TextField>

              <Box sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Signal-to-Noise Ratio (SNR): {snr} dB
                </Typography>
                <Slider
                  value={snr}
                  min={-10}
                  max={30}
                  step={1}
                  valueLabelDisplay="auto"
                  onChange={(_e, val) => setSnr(val as number)}
                />
              </Box>

              <TextField
                select
                label="Target ML/DL Classifier"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                fullWidth
              >
                <ListSubheader
                  sx={{
                    fontWeight: "bold",
                    color: "primary.main",
                    bgcolor: "background.paper",
                  }}
                >
                  Machine Learning Models
                </ListSubheader>
                {availableMLModels.map((model) => {
                  const displayNames: Record<string, string> = {
                    randomforest: "Random Forest (Classical)",
                    svm: "Support Vector Machine (Classical)",
                    gradientboosting: "Gradient Boosting (Classical)",
                    xgboost: "XGBoost (Classical)",
                    lightgbm: "LightGBM (Classical)",
                    knn: "K-Nearest Neighbors (Classical)",
                    logisticregression: "Logistic Regression (Classical)",
                  };
                  return (
                    <MenuItem key={model} value={model} sx={{ pl: 3 }}>
                      {displayNames[model] || model.toUpperCase()}
                    </MenuItem>
                  );
                })}

                <ListSubheader
                  sx={{
                    fontWeight: "bold",
                    color: "primary.main",
                    bgcolor: "background.paper",
                  }}
                >
                  Deep Learning Models
                </ListSubheader>
                {availableDLModels.map((model) => {
                  const displayNames: Record<string, string> = {
                    cnn1d: "1D CNN (Deep Learning)",
                    cnnlstm: "CNN-LSTM (Deep Learning)",
                    cnn2d: "2D CNN Spectrogram (Deep Learning)",
                  };
                  return (
                    <MenuItem key={model} value={model} sx={{ pl: 3 }}>
                      {displayNames[model] || model.toUpperCase()}
                    </MenuItem>
                  );
                })}
              </TextField>

              <TextField
                select
                label="Sample Rate (Hz)"
                value={sampleRate}
                onChange={(e) => setSampleRate(Number(e.target.value))}
                fullWidth
              >
                <MenuItem value={500000}>500 kHz</MenuItem>
                <MenuItem value={1000000}>1.0 MHz</MenuItem>
                <MenuItem value={2000000}>2.0 MHz</MenuItem>
              </TextField>

              <TextField
                select
                label="Number of Samples"
                value={numSamples}
                onChange={(e) => setNumSamples(Number(e.target.value))}
                fullWidth
              >
                <MenuItem value={256}>256 Samples</MenuItem>
                <MenuItem value={512}>512 Samples</MenuItem>
                <MenuItem value={1024}>1024 Samples</MenuItem>
              </TextField>
            </CardContent>
          </Card>

          {/* Impairments Panel */}
          <Accordion
            variant="outlined"
            sx={{ borderRadius: 2, overflow: "hidden" }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ fontWeight: 700 }}>
                2. Channel Impairments
              </Typography>
            </AccordionSummary>
            <AccordionDetails
              sx={{ display: "flex", flexDirection: "column", gap: 2 }}
            >
              <Grid container spacing={2}>
                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={awgnEnabled}
                        onChange={(e) => setAwgnEnabled(e.target.checked)}
                      />
                    }
                    label="AWGN Channel Noise"
                  />
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={rayleighEnabled}
                        onChange={(e) => setRayleighEnabled(e.target.checked)}
                      />
                    }
                    label="Rayleigh Fading"
                  />
                  {rayleighEnabled && (
                    <Box sx={{ pl: 2, mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Doppler Frequency: {rayleighDoppler} Hz
                      </Typography>
                      <Slider
                        value={rayleighDoppler}
                        min={0.1}
                        max={100}
                        step={0.5}
                        onChange={(_e, val) =>
                          setRayleighDoppler(val as number)
                        }
                      />
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={ricianEnabled}
                        onChange={(e) => setRicianEnabled(e.target.checked)}
                      />
                    }
                    label="Rician Fading"
                  />
                  {ricianEnabled && (
                    <Box
                      sx={{
                        pl: 2,
                        mt: 1,
                        display: "flex",
                        flexDirection: "column",
                        gap: 1.5,
                      }}
                    >
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          K Factor: {ricianK}
                        </Typography>
                        <Slider
                          value={ricianK}
                          min={1}
                          max={20}
                          step={0.5}
                          onChange={(_e, val) => setRicianK(val as number)}
                        />
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Doppler Frequency: {ricianDoppler} Hz
                        </Typography>
                        <Slider
                          value={ricianDoppler}
                          min={0.1}
                          max={50}
                          step={0.5}
                          onChange={(_e, val) =>
                            setRicianDoppler(val as number)
                          }
                        />
                      </Box>
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={freqOffsetEnabled}
                        onChange={(e) => setFreqOffsetEnabled(e.target.checked)}
                      />
                    }
                    label="Frequency Offset"
                  />
                  {freqOffsetEnabled && (
                    <Box sx={{ pl: 2, mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Offset: {freqOffset} Hz
                      </Typography>
                      <Slider
                        value={freqOffset}
                        min={-500}
                        max={500}
                        step={10}
                        onChange={(_e, val) => setFreqOffset(val as number)}
                      />
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={phaseNoiseEnabled}
                        onChange={(e) => setPhaseNoiseEnabled(e.target.checked)}
                      />
                    }
                    label="Phase Noise"
                  />
                  {phaseNoiseEnabled && (
                    <Box sx={{ pl: 2, mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Std Deviation: {phaseNoiseStd} rad
                      </Typography>
                      <Slider
                        value={phaseNoiseStd}
                        min={0.01}
                        max={0.5}
                        step={0.01}
                        onChange={(_e, val) => setPhaseNoiseStd(val as number)}
                      />
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={iqImbalanceEnabled}
                        onChange={(e) =>
                          setIqImbalanceEnabled(e.target.checked)
                        }
                      />
                    }
                    label="IQ Imbalance"
                  />
                  {iqImbalanceEnabled && (
                    <Box
                      sx={{
                        pl: 2,
                        mt: 1,
                        display: "flex",
                        flexDirection: "column",
                        gap: 1.5,
                      }}
                    >
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Amplitude Imbalance: {iqAmpImb} dB
                        </Typography>
                        <Slider
                          value={iqAmpImb}
                          min={0}
                          max={3}
                          step={0.1}
                          onChange={(_e, val) => setIqAmpImb(val as number)}
                        />
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Phase Imbalance: {iqPhaseImb}°
                        </Typography>
                        <Slider
                          value={iqPhaseImb}
                          min={0}
                          max={15}
                          step={0.5}
                          onChange={(_e, val) => setIqPhaseImb(val as number)}
                        />
                      </Box>
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={timingOffsetEnabled}
                        onChange={(e) =>
                          setTimingOffsetEnabled(e.target.checked)
                        }
                      />
                    }
                    label="Timing Offset"
                  />
                  {timingOffsetEnabled && (
                    <Box sx={{ pl: 2, mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Fractional Delay: {timingOffset}
                      </Typography>
                      <Slider
                        value={timingOffset}
                        min={0}
                        max={1}
                        step={0.05}
                        onChange={(_e, val) => setTimingOffset(val as number)}
                      />
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={multipathEnabled}
                        onChange={(e) => setMultipathEnabled(e.target.checked)}
                      />
                    }
                    label="Multipath Fading"
                  />
                  {multipathEnabled && (
                    <Box
                      sx={{
                        pl: 2,
                        mt: 1,
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                      }}
                    >
                      <TextField
                        size="small"
                        label="Delays (comma-separated)"
                        value={multipathDelays}
                        onChange={(e) => setMultipathDelays(e.target.value)}
                        fullWidth
                      />
                      <TextField
                        size="small"
                        label="Gains (dB, comma-separated)"
                        value={multipathGains}
                        onChange={(e) => setMultipathGains(e.target.value)}
                        fullWidth
                      />
                    </Box>
                  )}
                </Grid>

                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={clockDriftEnabled}
                        onChange={(e) => setClockDriftEnabled(e.target.checked)}
                      />
                    }
                    label="Clock Drift"
                  />
                  {clockDriftEnabled && (
                    <Box sx={{ pl: 2, mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Drift: {clockDrift} PPM
                      </Typography>
                      <Slider
                        value={clockDrift}
                        min={0}
                        max={100}
                        step={1}
                        onChange={(_e, val) => setClockDrift(val as number)}
                      />
                    </Box>
                  )}
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Button
            variant="contained"
            color="primary"
            fullWidth
            size="large"
            onClick={handleSimulateAndPredict}
            disabled={loading}
            startIcon={
              loading ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <PlayIcon />
              )
            }
          >
            {loading ? "Processing Predictor..." : "Predict Modulation"}
          </Button>
        </Grid>

        {/* Right Side: Visualizations and Decision Outputs */}
        <Grid size={{ xs: 12, lg: 8 }}>
          {loading ? (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <LoadingSkeleton variant="stats" count={2} />
              <LoadingSkeleton variant="plot" height={420} />
            </Box>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {/* Prediction Latency & Results Banner */}
              {prediction ? (
                <Card
                  sx={{
                    background: (theme) =>
                      theme.palette.mode === "dark"
                        ? "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
                        : "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)",
                    color: "text.primary",
                    border: "none",
                    borderLeft: "5px solid",
                    borderLeftColor: "primary.main",
                    boxShadow: (theme) =>
                      theme.palette.mode === "dark"
                        ? "none"
                        : "0 4px 20px 0 rgba(0,0,0,0.05)",
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Grid container spacing={3} sx={{ alignItems: "center" }}>
                      <Grid size={{ xs: 12, sm: 6 }}>
                        <Typography
                          variant="overline"
                          sx={{
                            letterSpacing: 1,
                            color: (theme) =>
                              theme.palette.mode === "dark"
                                ? "primary.light"
                                : "primary.main",
                            fontWeight: 600,
                          }}
                        >
                          CLASSIFICATION SUMMARY ({selectedModel.toUpperCase()})
                        </Typography>
                        <Typography
                          variant="h3"
                          sx={{ fontWeight: 900, mt: 1 }}
                        >
                          {prediction.predicted_class}
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{ opacity: 0.8, mt: 1 }}
                        >
                          Actual Configured: <strong>{modulation}</strong>
                        </Typography>
                      </Grid>

                      <Grid
                        size={{ xs: 12, sm: 6 }}
                        sx={{
                          textAlign: { sm: "right" },
                          display: "flex",
                          justifyContent: { xs: "flex-start", sm: "flex-end" },
                          gap: 3,
                          alignItems: "center",
                        }}
                      >
                        {inferenceTime !== null && (
                          <Box sx={{ textAlign: "center" }}>
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                gap: 0.5,
                              }}
                            >
                              <TimerIcon
                                fontSize="small"
                                sx={{
                                  color: (theme) =>
                                    theme.palette.mode === "dark"
                                      ? "primary.light"
                                      : "primary.main",
                                }}
                              />
                              <Typography variant="h5" sx={{ fontWeight: 800 }}>
                                {inferenceTime} ms
                              </Typography>
                            </Box>
                            <Typography variant="caption" sx={{ opacity: 0.7 }}>
                              Inference Time
                            </Typography>
                          </Box>
                        )}

                        <Box sx={{ textAlign: "center" }}>
                          <Box
                            sx={{
                              display: "inline-block",
                              position: "relative",
                            }}
                          >
                            <CircularProgress
                              variant="determinate"
                              value={prediction.confidence * 100}
                              size={70}
                              thickness={6}
                              sx={{
                                color:
                                  prediction.predicted_class === modulation
                                    ? "success.main"
                                    : "warning.main",
                              }}
                            />
                            <Box
                              sx={{
                                top: 0,
                                left: 0,
                                bottom: 0,
                                right: 0,
                                position: "absolute",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                              }}
                            >
                              <Typography
                                variant="subtitle2"
                                component="div"
                                sx={{ fontWeight: "bold" }}
                              >
                                {Math.round(prediction.confidence * 100)}%
                              </Typography>
                            </Box>
                          </Box>
                          <Typography
                            variant="caption"
                            sx={{ display: "block", mt: 0.5, opacity: 0.8 }}
                          >
                            Confidence
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Probability bars */}
                    {prediction.probabilities && (
                      <Box sx={{ mt: 3 }}>
                        <Typography
                          variant="subtitle2"
                          sx={{
                            mb: 1.5,
                            fontWeight: 600,
                            color: (theme) =>
                              theme.palette.mode === "dark"
                                ? "primary.light"
                                : "primary.main",
                          }}
                        >
                          Classifier Probabilities:
                        </Typography>
                        {Object.entries(prediction.probabilities).map(
                          ([className, prob]) => (
                            <Box key={className} sx={{ mb: 1 }}>
                              <Box
                                sx={{
                                  display: "flex",
                                  justifyContent: "space-between",
                                  mb: 0.5,
                                }}
                              >
                                <Typography
                                  variant="caption"
                                  sx={{ fontWeight: 500 }}
                                >
                                  {className}
                                </Typography>
                                <Typography variant="caption">
                                  {Math.round(prob * 100)}%
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={prob * 100}
                                sx={{
                                  height: 6,
                                  borderRadius: 3,
                                  bgcolor: (theme) =>
                                    theme.palette.mode === "dark"
                                      ? "rgba(255,255,255,0.1)"
                                      : "rgba(0,0,0,0.05)",
                                  "& .MuiLinearProgress-bar": {
                                    bgcolor:
                                      className === modulation
                                        ? "success.main"
                                        : "primary.main",
                                  },
                                }}
                              />
                            </Box>
                          ),
                        )}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              ) : null}

              {/* Visualization panels */}
              <Card sx={{ minHeight: 400 }}>
                <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
                  <Tabs
                    value={activeTab}
                    onChange={handleTabChange}
                    sx={{ px: 2 }}
                  >
                    <Tab
                      icon={<ImageIcon />}
                      label="Generated Visualizations"
                      iconPosition="start"
                    />
                    <Tab
                      icon={<ListIcon />}
                      label="Feature Values"
                      iconPosition="start"
                    />
                  </Tabs>
                </Box>

                {/* Diagnostic Plots Tab */}
                {activeTab === 0 && (
                  <CardContent>
                    {plots.length > 0 ? (
                      <Grid container spacing={2}>
                        {plots.map((plot) => (
                          <Grid size={{ xs: 12, sm: 6 }} key={plot.name}>
                            <Card
                              variant="outlined"
                              sx={{ overflow: "hidden", height: "100%" }}
                            >
                              <Box
                                sx={{
                                  p: 1.5,
                                  bgcolor: "background.default",
                                  borderBottom: "1px solid",
                                  borderColor: "divider",
                                }}
                              >
                                <Typography
                                  variant="subtitle2"
                                  sx={{
                                    textTransform: "capitalize",
                                    fontWeight: 700,
                                  }}
                                >
                                  {plot.name.replace("_", " ")}
                                </Typography>
                              </Box>
                              <Box
                                sx={{
                                  display: "flex",
                                  justifyContent: "center",
                                  alignItems: "center",
                                  p: 1.5,
                                  minHeight: 200,
                                }}
                              >
                                <img
                                  src={`${API_BASE_URL}${plot.url}`}
                                  alt={plot.name}
                                  style={{
                                    maxWidth: "100%",
                                    height: "auto",
                                    borderRadius: 4,
                                  }}
                                />
                              </Box>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    ) : (
                      <EmptyState
                        title="No Signal Visualizations"
                        description="Configure the simulation parameters and impairments in the left panel, then hit Predict Modulation to generate signals and render constellation diagrams, spectral plots, or waveforms."
                        icon={<AnalyticsIcon sx={{ fontSize: 60 }} />}
                      />
                    )}
                  </CardContent>
                )}

                {/* Features Tab */}
                {activeTab === 1 && (
                  <CardContent>
                    {prediction ? (
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead sx={{ bgcolor: "background.default" }}>
                            <TableRow>
                              <TableCell sx={{ fontWeight: "bold" }}>
                                Statistical Parameter
                              </TableCell>
                              <TableCell
                                align="right"
                                sx={{ fontWeight: "bold" }}
                              >
                                Value
                              </TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(prediction.features).map(
                              ([key, val]) => (
                                <TableRow key={key} hover>
                                  <TableCell component="th" scope="row">
                                    <code>{key}</code>
                                  </TableCell>
                                  <TableCell align="right">
                                    {typeof val === "number"
                                      ? val.toFixed(6)
                                      : val}
                                  </TableCell>
                                </TableRow>
                              ),
                            )}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    ) : (
                      <EmptyState
                        title="No Statistical Features Compiled"
                        description="Run the modulation classifier simulation model first. The extracted statistical parameter table (e.g. HOC Cumulants, Phase variance) will automatically compile here."
                        icon={<ListIcon sx={{ fontSize: 60 }} />}
                      />
                    )}
                  </CardContent>
                )}
              </Card>
            </Box>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default AmcPredictor;
