import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  Alert,
  Divider,
  Paper,
  CardActionArea,
} from '@mui/material';
import {
  BarChart as ChartIcon,
  CheckCircle as CheckedIcon,
  ErrorOutlined as ErrorIcon,
  Storage as StorageIcon,
  FilterAlt as FeaturesIcon,
  Psychology as MLIcon,
  SettingsSuggest as ConfigIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { healthApi } from '../api/healthApi';
import { resultsApi } from '../api/resultsApi';
import { plotsApi } from '../api/plotsApi';
import type { HealthResponse } from '../api/healthApi';
import type { EvaluationResults } from '../api/resultsApi';
import type { PlotFile } from '../api/plotsApi';
import AcademicHeader from '../components/Common/AcademicHeader';
import LoadingSkeleton from '../components/Common/LoadingSkeleton';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // States for backend data
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthDL, setHealthDL] = useState<HealthResponse | null>(null);
  const [results, setResults] = useState<EvaluationResults | null>(null);
  const [plots, setPlots] = useState<PlotFile[]>([]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthRes, healthDLRes, resultsRes, plotsRes] = await Promise.all([
        healthApi.getHealth().catch((err) => {
          console.warn('Failed to load health status of ML models', err);
          return null;
        }),
        healthApi.getHealthDL().catch((err) => {
          console.warn('Failed to load health status of DL models', err);
          return null;
        }),
        resultsApi.getResults().catch((err) => {
          console.warn('Failed to load evaluation results', err);
          return null;
        }),
        plotsApi.getPlots().catch((err) => {
          console.warn('Failed to load plots', err);
          return [];
        }),
      ]);
      setHealth(healthRes);
      setHealthDL(healthDLRes);
      setResults(resultsRes);
      setPlots(plotsRes);
    } catch (err: any) {
      setError(
        err?.message ||
          'Could not connect to the backend REST API. Please ensure the FastAPI server is running on port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Determine dynamic stats and status values
  const isBackendOnline = health !== null || healthDL !== null;
  
  const numMlModelsLoaded = health
    ? Object.values(health.models_loaded).filter(Boolean).length
    : 0;
  const numDlModelsLoaded = healthDL
    ? Object.values(healthDL.models_loaded).filter(Boolean).length
    : 0;
  const numModelsLoaded = numMlModelsLoaded + numDlModelsLoaded;

  const totalMlModelsCount = health ? Object.keys(health.models_loaded).length : 7;
  const totalDlModelsCount = healthDL ? Object.keys(healthDL.models_loaded).length : 3;
  const totalModelsCount = totalMlModelsCount + totalDlModelsCount;

  const numMlOutputs = results?.available_files?.length || 0;
  const numGeneratedPlots = plots.filter((p) => p.category === 'signal').length;

  // Best model detection
  let bestModelName = 'N/A';
  let bestModelAcc = 0;
  if (results?.summary && results.summary.length > 0) {
    results.summary.forEach((modelRow) => {
      const accKey = Object.keys(modelRow).find((k) => k.toLowerCase().includes('accuracy'));
      if (accKey) {
        const accVal = parseFloat(modelRow[accKey]);
        if (accVal > bestModelAcc) {
          bestModelAcc = accVal;
          bestModelName = modelRow['model'] || modelRow['Model'] || modelRow['Classifier'] || modelRow['classifier'] || 'Unknown';
        }
      }
    });
  }

  // Grouped project phase configurations
  const phases = [
    {
      title: 'Phase 1: SDR Dataset Generation',
      subtitle: 'Simulate high-fidelity baseband signals & channels',
      status: isBackendOnline ? 'Active' : 'Pending',
      statusColor: isBackendOnline ? 'success' : 'warning',
      outputsCount: numGeneratedPlots > 0 ? `${numGeneratedPlots} Plots Generated` : '6,300 Raw Waveforms Simulated',
      metadata: isBackendOnline ? 'Transmitter Config: BPSK, QPSK, QAM, OFDM, AM, FM' : 'Server Connection Needed',
      icon: <StorageIcon color="primary" sx={{ fontSize: 40 }} />,
      targetPath: '/results',
      state: { tab: 0 },
    },
    {
      title: 'Phase 2: Feature Extraction',
      subtitle: 'Extract envelope, spectral, and HOC cumulant features',
      status: results?.summary ? 'Complete' : 'Pending',
      statusColor: results?.summary ? 'success' : 'warning',
      outputsCount: '39 Features Extracted',
      metadata: 'Features: papr, inst_phase_std, cumulants C42, spectral_flatness',
      icon: <FeaturesIcon color="primary" sx={{ fontSize: 40 }} />,
      targetPath: '/results',
      state: { tab: 1 },
    },
    {
      title: 'Phase 3: Feature Analysis & Ranking',
      subtitle: 'Perform dimensionality reduction & selection',
      status: isBackendOnline ? 'Active' : 'Pending',
      statusColor: isBackendOnline ? 'success' : 'warning',
      outputsCount: 'PCA / t-SNE / UMAP Active',
      metadata: 'Variance thresholding & classification ranking active',
      icon: <ConfigIcon color="primary" sx={{ fontSize: 40 }} />,
      targetPath: '/results',
      state: { tab: 2 },
    },
    {
      title: 'Phase 4: ML Classifier Training',
      status: results && results.summary && results.summary.length > 0 ? 'Complete' : 'Pending',
      statusColor: results && results.summary && results.summary.length > 0 ? 'success' : 'warning',
      subtitle: 'Train classical machine learning classifiers',
      outputsCount: results?.summary ? `${results.summary.length} Classifiers Trained` : '0 Classifiers Trained',
      metadata: bestModelAcc > 0 ? `Best Model: ${bestModelName} (${(bestModelAcc * 100).toFixed(1)}% Acc)` : 'Model Performance File Missing',
      icon: <MLIcon color="primary" sx={{ fontSize: 40 }} />,
      targetPath: '/performance',
      state: { tab: 0 },
    },
    {
      title: 'Phase 5: Deep Learning Inference',
      status: numDlModelsLoaded > 0 ? 'Active' : 'Pending',
      statusColor: numDlModelsLoaded > 0 ? 'success' : 'warning',
      subtitle: 'Evaluate and run high-dimensional CNN/LSTM neural models',
      outputsCount: `${numDlModelsLoaded} / ${totalDlModelsCount} Models Ready`,
      metadata: `Active network architectures: CNN1D, CNNLSTM, CNN2D`,
      icon: <ChartIcon color="primary" sx={{ fontSize: 40 }} />,
      targetPath: '/results',
      state: { tab: 4 },
    },
  ];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AcademicHeader
        title="Automatic Modulation Classification Dashboard"
        subtitle="Full-cycle baseband simulator, impairing channel emulator, and ML/DL modulation classification platform."
        action={
          <Button
            variant="outlined"
            color="primary"
            onClick={fetchData}
            disabled={loading}
            startIcon={<RefreshIcon />}
          >
            Refresh
          </Button>
        }
      />

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <>
          <LoadingSkeleton variant="stats" count={4} />
          <Typography variant="h5" sx={{ fontWeight: 800, mb: 3 }}>
            Project Phases
          </Typography>
          <LoadingSkeleton variant="card" count={5} />
        </>
      ) : (
        <>
          {/* Quick Statistics Overview */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                  Backend Connection
                </Typography>
                <Box sx={{ mt: 1.5 }}>
                  <Chip
                    icon={isBackendOnline ? <CheckedIcon /> : <ErrorIcon />}
                    label={isBackendOnline ? 'Online' : 'Offline'}
                    color={isBackendOnline ? 'success' : 'error'}
                    sx={{ fontWeight: 'bold' }}
                  />
                </Box>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'left', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                  Supported Models: <Box component="span" color="primary.main" sx={{ fontWeight: 800 }}>{totalModelsCount}</Box>
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 1 }}>
                  Currently Loaded: <Box component="span" color="primary.main" sx={{ fontWeight: 800 }}>{numModelsLoaded}</Box>
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  ML Loaded: <Box component="span" color="primary.main" sx={{ fontWeight: 700 }}>{numMlModelsLoaded}/7</Box>
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  DL Loaded: <Box component="span" color="primary.main" sx={{ fontWeight: 700 }}>{numDlModelsLoaded}/3 (Lazy Loaded)</Box>
                </Typography>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                  Evaluation Assets
                </Typography>
                <Typography variant="h4" color="primary" sx={{ fontWeight: 800, mt: 0.5 }}>
                  {numMlOutputs} files
                </Typography>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                  Best Classifier
                </Typography>
                <Typography variant="h5" color="primary" sx={{ fontWeight: 800, mt: 1, textTransform: 'capitalize' }}>
                  {bestModelName}
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          {/* Five Implementation Phases */}
          <Typography variant="h5" sx={{ fontWeight: 800, mb: 3 }}>
            Project Phases
          </Typography>

          <Grid container spacing={3} sx={{ mb: 4 }}>
            {phases.map((phase, idx) => (
              <Grid size={{ xs: 12, md: 6, lg: 4 }} key={idx}>
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', transition: 'transform 0.2s, box-shadow 0.2s', '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 } }}>
                  <CardActionArea sx={{ flexGrow: 1, p: 1 }} onClick={() => navigate(phase.targetPath, { state: phase.state })}>
                    <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        {phase.icon}
                        <Chip
                          label={phase.status}
                          color={phase.statusColor as any}
                          size="small"
                          sx={{ fontWeight: 'bold' }}
                        />
                      </Box>

                      <Typography variant="h6" sx={{ fontWeight: 800, mb: 1 }}>
                        {phase.title}
                      </Typography>

                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                        {phase.subtitle}
                      </Typography>

                      <Divider sx={{ my: 1.5 }} />

                      <Box>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 600 }}>
                          Outputs: {phase.outputsCount}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                          Metadata: {phase.metadata}
                        </Typography>
                      </Box>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}
    </Box>
  );
};

export default Dashboard;
