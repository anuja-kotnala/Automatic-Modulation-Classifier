import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  CheckCircleOutlined as CheckIcon,
  Memory as ChipIcon,
  Storage as DbIcon,
  Code as CodeIcon,
  Timeline as PipelineIcon,
} from '@mui/icons-material';
import AcademicHeader from '../components/Common/AcademicHeader';

export const About: React.FC = () => {
  return (
    <Box sx={{ maxWidth: 1000, mx: 'auto', py: 2 }}>
      <AcademicHeader
        title="About The AMC Project"
        subtitle="Academic documentation describing the architecture, ML/DL models, baseband signal generator details, and technologies driving this framework."
      />

      {/* 1. Project Objective */}
      <Paper variant="outlined" sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckIcon color="primary" /> Project Objective
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          The core objective of this project is to develop and evaluate a robust, configuration-driven workbench for <strong>Automatic Modulation Classification (AMC)</strong>. By integrating complex baseband generation pipelines with fading channel impairment emulators, this project simulates realistic software-defined radio (SDR) environments. It enables researchers to analyze and compare feature-engineered classifiers (Random Forest, SVM, XGBoost) and raw IQ deep learning neural networks (CNN1D, CNN-LSTM, CNN2D) under high-order impairment conditions.
        </Typography>
      </Paper>

      {/* 2. Overview of AMC */}
      <Paper variant="outlined" sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
          What is Automatic Modulation Classification?
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          In modern cognitive radio systems and spectrum monitoring setups, receiver devices often detect signals without any prior negotiation or control channel handshake. AMC is the algorithmic process that automatically identifies the modulation scheme (e.g., BPSK, QPSK, 16QAM, 64QAM, AM, FM, OFDM) of a received radio carrier.
        </Typography>
        <Typography variant="body1" color="text.secondary">
          This classification plays a vital role in <strong>interference source localization, military electronic warfare, dynamic spectrum access, and signal intelligence</strong>. The challenge lies in extracting distinguishing statistical cumulants (high-order moments) or spectral features from the incoming IQ waveforms in the presence of noise, frequency offsets, phase jitter, and multi-path fading.
        </Typography>
      </Paper>

      {/* 3. Project Pipeline */}
      <Paper variant="outlined" sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <PipelineIcon color="primary" /> Signal Processing & ML Pipeline
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 4 }}>
            <Card variant="outlined" sx={{ height: '100%', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
              <CardContent>
                <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 'bold', mb: 1 }}>
                  1. Waveform Generation
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Complex baseband simulation of analog (AM, FM) and digital (BPSK, QPSK, 16QAM, 64QAM, OFDM) signals using custom OOP modulators.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid size={{ xs: 12, md: 4 }}>
            <Card variant="outlined" sx={{ height: '100%', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
              <CardContent>
                <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 'bold', mb: 1 }}>
                  2. Channel Impairments
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Wireless channel emulation adding AWGN, Doppler Rayleigh/Rician multipath fading, phase noise, clock drifts, fractional timing offsets, and IQ imbalances.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid size={{ xs: 12, md: 4 }}>
            <Card variant="outlined" sx={{ height: '100%', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
              <CardContent>
                <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 'bold', mb: 1 }}>
                  3. Extraction & Training
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Compiles 39 envelope, spectral, and HOC cumulant features (C42, kurtosis, papr) to fit ML models (Random Forest, SVM, LightGBM) and deep networks.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* 4. Deep Learning Models */}
      <Paper variant="outlined" sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <ChipIcon color="primary" /> Deep Learning Classifier Models
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          The deep learning subsystem features multiple custom neural network architectures implemented in PyTorch:
        </Typography>
        <List dense>
          <ListItem disablePadding sx={{ mb: 1.5 }}>
            <ListItemIcon><CheckIcon color="secondary" fontSize="small" /></ListItemIcon>
            <ListItemText
              primary="CNN1D (1D Convolutional Neural Network)"
              secondary="Accepts raw 2 x N complex float tensors directly representing In-Phase and Quadrature components. Applies deep 1D convolutions, batch normalization, max pooling, and dense decision layers directly on IQ sequences."
            />
          </ListItem>
          <ListItem disablePadding sx={{ mb: 1.5 }}>
            <ListItemIcon><CheckIcon color="secondary" fontSize="small" /></ListItemIcon>
            <ListItemText
              primary="CNNLSTM (CNN + Long Short-Term Memory)"
              secondary="Performs initial feature extraction using 1D convolutional layers, followed by bidirectional LSTM sequence modelling to capture complex temporal dependency dynamics in raw IQ wave streams."
            />
          </ListItem>
          <ListItem disablePadding>
            <ListItemIcon><CheckIcon color="secondary" fontSize="small" /></ListItemIcon>
            <ListItemText
              primary="CNN2D (2D Spectrogram Convolutional Network)"
              secondary="Transforms raw IQ signals into centered 2D power spectrogram images, classification is then performed utilizing 2D spatial convolutions, batch normalization, and adaptive pooling."
            />
          </ListItem>
        </List>
      </Paper>

      {/* 5. System Architecture */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Backend Architecture */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper variant="outlined" sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <DbIcon color="primary" /> Backend REST Architecture
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              The backend layer is a high-performance REST API built using <strong>FastAPI (Python 3)</strong>. It encapsulates:
            </Typography>
            <List dense>
              <ListItem><ListItemText primary="Uvicorn ASGI runner on port 8000" /></ListItem>
              <ListItem><ListItemText primary="Pydantic schemas validating simulation parameters" /></ListItem>
              <ListItem><ListItemText primary="Joblib serialization loading ML classifiers on-demand" /></ListItem>
              <ListItem><ListItemText primary="Matplotlib visualization modules saving diagnostic charts" /></ListItem>
              <ListItem><ListItemText primary="NumPy & Pandas handling IQ tensor computations" /></ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Frontend Architecture */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper variant="outlined" sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <CodeIcon color="primary" /> Frontend Architecture
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              A responsive single-page web app built on **React 18** and **Vite (TypeScript)**:
            </Typography>
            <List dense>
              <ListItem><ListItemText primary="Material UI (MUI v6) UI components and grids" /></ListItem>
              <ListItem><ListItemText primary="React Router v6 declarative routing" /></ListItem>
              <ListItem><ListItemText primary="Axios client centralizing endpoints with retry logic" /></ListItem>
              <ListItem><ListItemText primary="Theme toggle context serving Light & Dark palettes" /></ListItem>
              <ListItem><ListItemText primary="Modal preview dialogues supporting scale zoom controls" /></ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* 6. Future Improvements */}
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
          Future Improvements
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Planned additions to enhance the AMC research platform include:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemIcon><CheckIcon color="primary" fontSize="small" /></ListItemIcon>
            <ListItemText primary="Support for SigMF data imports (standard SDR capture formatting)." />
          </ListItem>
          <ListItem>
            <ListItemIcon><CheckIcon color="primary" fontSize="small" /></ListItemIcon>
            <ListItemText primary="Adding Transformer-based classification models (Self-Attention networks on raw IQ)." />
          </ListItem>
          <ListItem>
            <ListItemIcon><CheckIcon color="primary" fontSize="small" /></ListItemIcon>
            <ListItemText primary="Adding real-time streaming spectrum visualizers using WebSockets." />
          </ListItem>
        </List>
      </Paper>
    </Box>
  );
};

export default About;
