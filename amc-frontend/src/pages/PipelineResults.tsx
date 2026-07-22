import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Box,
  Tabs,
  Tab,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  Dialog,
  DialogContent,
  DialogActions,
  IconButton,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Divider,
} from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Fullscreen as FullscreenIcon,
  Download as DownloadIcon,
  Close as CloseIcon,
  TableChart as TableIcon,
  InsertPhoto as PhotoIcon,
} from '@mui/icons-material';
import { amcApi } from '../services/amcApi';
import type { PlotFile } from '../services/amcApi';
import { API_BASE_URL } from '../api/client';
import AcademicHeader from '../components/Common/AcademicHeader';
import LoadingSkeleton from '../components/Common/LoadingSkeleton';
import EmptyState from '../components/Common/EmptyState';

const TABLE_COLUMNS = [
  { key: 'model_name', label: 'Model Name' },
  { key: 'model_type', label: 'Model Type' },
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'f1_score', label: 'F1 Score' },
  { key: 'cv_mean', label: 'CV Mean' },
  { key: 'cv_std', label: 'CV Std' },
  { key: 'macro_auc', label: 'Macro AUC' },
  { key: 'parameters', label: 'Parameters' },
  { key: 'training_time', label: 'Training Time (s)' },
  { key: 'avg_epoch_time', label: 'Avg Epoch Time (s)' },
  { key: 'model_size_mb', label: 'Model Size (MB)' },
  { key: 'best_val_accuracy', label: 'Best Val Acc' },
  { key: 'best_val_loss', label: 'Best Val Loss' },
  { key: 'inference_time_ms', label: 'Inference Time (ms)' },
];

export const PipelineResults: React.FC = () => {
  const location = useLocation();
  const initialTab = location.state && typeof location.state.tab === 'number' ? location.state.tab : 0;
  const [activeTab, setActiveTab] = useState(initialTab);

  useEffect(() => {
    if (location.state && typeof location.state.tab === 'number') {
      setActiveTab(location.state.tab);
    }
  }, [location.state]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [plots, setPlots] = useState<PlotFile[]>([]);
  const [featureRankings, setFeatureRankings] = useState<Array<Record<string, string>>>([]);
  const [mlSummary, setMlSummary] = useState<Array<Record<string, any>>>([]);
  const [dlModels, setDlModels] = useState<string[]>(['cnn1d', 'cnnlstm', 'cnn2d']);

  // Preview Dialog states
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewImgUrl, setPreviewImgUrl] = useState('');
  const [previewTitle, setPreviewTitle] = useState('');
  const [zoomScale, setZoomScale] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const loadPipelineData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [plotsRes, rankingsRes, resultsRes, dlHealthRes] = await Promise.all([
        amcApi.getPlots().catch(() => []),
        amcApi.getFeatureRankings().catch(() => []),
        amcApi.getResults().catch(() => ({ summary: [], available_files: [] })),
        amcApi.getHealthDL().catch(() => null),
      ]);

      setPlots(plotsRes);
      setFeatureRankings(rankingsRes);
      setMlSummary(resultsRes.summary || []);
      if (dlHealthRes && dlHealthRes.models_loaded) {
        setDlModels(Object.keys(dlHealthRes.models_loaded));
      }
    } catch (err: any) {
      setError('Could not fetch pipeline products. Ensure the backend APIs are online.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPipelineData();
  }, []);

  const openPreview = (url: string, title: string) => {
    setPreviewImgUrl(url);
    setPreviewTitle(title);
    setZoomScale(1);
    setIsFullscreen(false);
    setPreviewOpen(true);
  };

  const closePreview = () => {
    setPreviewOpen(false);
    setPreviewImgUrl('');
    setPreviewTitle('');
  };

  const handleZoomIn = () => setZoomScale((prev) => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoomScale((prev) => Math.max(prev - 0.25, 0.5));

  const handleDownload = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Grouped products per phase
  const signalPlots = plots.filter((p) => p.category === 'signal');
  const mlEvalPlots = plots.filter((p) => p.category === 'evaluation');

  const phase3Plots = [
    { name: 'PCA Dimension Reduction', url: '/static/analysis/dim_reduction_pca.png' },
    { name: 't-SNE Dimension Reduction', url: '/static/analysis/dim_reduction_tsne.png' },
    { name: 'UMAP Dimension Reduction', url: '/static/analysis/dim_reduction_umap.png' },
    { name: 'Feature Correlation Matrix', url: '/static/analysis/correlation_matrix.png' },
    { name: 'Feature vs SNR Distribution', url: '/static/analysis/feature_vs_snr.png' },
    { name: 'PAPR Boxplot', url: '/static/analysis/papr_boxplot.png' },
    { name: 'PAPR Histogram', url: '/static/analysis/papr_histogram.png' },
    { name: 'PAPR Violin', url: '/static/analysis/papr_violin.png' },
    { name: 'SHAP Feature Quality Summary', url: '/static/analysis/shap_summary_plot.png' },
    { name: 'Top Features Bar Chart', url: '/static/analysis/top_ranked_features.png' },
  ];

  const getModelDisplayName = (modelId: string) => {
    const displayNames: Record<string, string> = {
      cnn1d: 'CNN1D (Raw IQ)',
      cnnlstm: 'CNN-LSTM (Raw IQ)',
      cnn2d: 'CNN2D Spectrogram'
    };
    return displayNames[modelId.toLowerCase().trim()] || modelId.toUpperCase();
  };

  const phase5Plots = dlModels.flatMap((model) => {
    const displayName = getModelDisplayName(model);
    return [
      { name: `${displayName} Confusion Matrix`, url: `/static/dl_eval/${model.toLowerCase().trim()}_raw_confusion_matrix.png` },
      { name: `${displayName} Learning Curves`, url: `/static/dl_eval/${model.toLowerCase().trim()}_raw_learning_curves.png` }
    ];
  });

  const renderProductCard = (title: string, url: string, filename: string) => {
    const fullUrl = `${API_BASE_URL}${url}`;
    return (
      <Grid size={{ xs: 12, sm: 6, md: 4 }} key={filename}>
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
          <Box sx={{ position: 'relative', pt: '56.25%', bgcolor: 'action.hover' }}>
            <CardMedia
              component="img"
              src={fullUrl}
              alt={title}
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                cursor: 'zoom-in',
              }}
              onClick={() => openPreview(fullUrl, title)}
            />
          </Box>
          <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>
              {title}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                fullWidth
                startIcon={<FullscreenIcon />}
                onClick={() => openPreview(fullUrl, title)}
              >
                Preview
              </Button>
              <Button
                variant="contained"
                size="small"
                fullWidth
                startIcon={<DownloadIcon />}
                onClick={() => handleDownload(fullUrl, filename)}
              >
                Download
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    );
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AcademicHeader
        title="Pipeline Results & Artifacts"
        subtitle="Browse diagnostic plots, dimensionality reduction projections, statistical lists, classifier accuracy, and deep learning metrics."
      />

      {error && <Alert severity="error" sx={{ mb: 4 }}>{error}</Alert>}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
          <Tab label="Phase 1: Dataset" />
          <Tab label="Phase 2: Features" />
          <Tab label="Phase 3: Analysis" />
          <Tab label="Phase 4: ML Classifiers" />
          <Tab label="Phase 5: Deep Learning" />
        </Tabs>
      </Box>

      {loading ? (
        <LoadingSkeleton variant="card" count={3} />
      ) : (
        <Box>
          {/* Phase 1 Tab */}
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                SDR Simulated Signals & Receiver Diagnostics
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Calculated constellations, waveforms, magnitude spectrums, spectrograms, waterfalls, and eye diagrams.
              </Typography>
              <Divider sx={{ mb: 3 }} />

              {signalPlots.length > 0 ? (
                <Grid container spacing={3}>
                  {signalPlots.map((plot) => renderProductCard(plot.name.replace('.png', '').replace(/_/g, ' '), plot.url, plot.name))}
                </Grid>
              ) : (
                <EmptyState
                  title="No Signal Plots Found"
                  description="Run a transmitter simulation on the AMC Predictor panel to generate signals."
                  icon={<PhotoIcon sx={{ fontSize: 60 }} />}
                />
              )}
            </Box>
          )}

          {/* Phase 2 Tab */}
          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                Computed Feature Set Rankings
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Features ranking list sorted by statistical distinctiveness in communications classifying.
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 7 }}>
                  <Card>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                        <TableIcon color="primary" /> Feature List Rankings (feature_rankings.csv)
                      </Typography>
                      {featureRankings.length > 0 ? (
                        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 500, borderRadius: 2 }}>
                          <Table stickyHeader size="small">
                            <TableHead>
                              <TableRow>
                                {Object.keys(featureRankings[0]).map((h) => (
                                  <TableCell key={h} sx={{ fontWeight: 'bold', bgcolor: 'background.default' }}>
                                    {h}
                                  </TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {featureRankings.map((row, idx) => (
                                <TableRow key={idx} hover>
                                  {Object.values(row).map((val, i) => (
                                    <TableCell key={i}>
                                      {isNaN(Number(val)) ? val : parseFloat(val).toFixed(5)}
                                    </TableCell>
                                  ))}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      ) : (
                        <EmptyState
                          title="Feature rankings unavailable"
                          description="The feature_rankings.csv file was not found on the backend."
                          icon={<TableIcon sx={{ fontSize: 60 }} />}
                        />
                      )}
                    </CardContent>
                  </Card>
                </Grid>
                <Grid size={{ xs: 12, md: 5 }}>
                  <Card>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                        <PhotoIcon color="primary" /> Feature Class Separation Plot
                      </Typography>
                      <Box sx={{ p: 1.5, bgcolor: 'background.default', display: 'flex', justifyContent: 'center', borderRadius: 2 }}>
                        <img
                          src={`${API_BASE_URL}/static/analysis/top_ranked_features.png`}
                          alt="Top Features Chart"
                          style={{ maxWidth: '100%', height: 'auto', borderRadius: 4, cursor: 'zoom-in' }}
                          onClick={() => openPreview(`${API_BASE_URL}/static/analysis/top_ranked_features.png`, 'Top Ranked Features Separation')}
                          onError={(e) => {
                            (e.target as any).style.display = 'none';
                          }}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}

          {/* Phase 3 Tab */}
          {activeTab === 2 && (
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                Dimensionality Reduction & Statistical Clusters
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Projections mapping high-dimensional communications parameters to 2D features subspaces (PCA, t-SNE, UMAP).
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <Grid container spacing={3}>
                {phase3Plots.map((plot) => renderProductCard(plot.name, plot.url, plot.url.substring(plot.url.lastIndexOf('/') + 1)))}
              </Grid>
            </Box>
          )}

          {/* Phase 4 Tab */}
          {activeTab === 3 && (
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                Machine Learning Classifiers Performance metrics
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Accuracies, precision scores, F1 ratings, and evaluation maps computed across models.
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <Card sx={{ mb: 4 }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TableIcon color="primary" /> Classifiers Accuracy Table
                  </Typography>
                  {mlSummary.length > 0 ? (
                    (() => {
                      const activeColumns = TABLE_COLUMNS.filter((col) =>
                        mlSummary.some((row) => row[col.key] !== undefined && row[col.key] !== null)
                      );
                      return (
                        <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                          <Table>
                            <TableHead sx={{ bgcolor: 'background.default' }}>
                              <TableRow>
                                {activeColumns.map((col) => (
                                  <TableCell key={col.key} sx={{ fontWeight: 'bold' }}>
                                    {col.label}
                                  </TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {mlSummary.map((row, idx) => (
                                <TableRow key={idx} hover>
                                  {activeColumns.map((col) => {
                                    const val = row[col.key];
                                    return (
                                      <TableCell key={col.key}>
                                        {val === undefined || val === null
                                          ? '-'
                                          : col.key === 'model_type'
                                          ? String(val).toUpperCase()
                                          : col.key === 'parameters'
                                          ? typeof val === 'number' ? val.toLocaleString() : String(val)
                                          : typeof val === 'number'
                                          ? val.toFixed(4)
                                          : String(val)}
                                      </TableCell>
                                    );
                                  })}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      );
                    })()
                  ) : (
                    <EmptyState
                      title="Accuracy Table Missing"
                      description="No summary CSV stats generated on the backend."
                      icon={<TableIcon sx={{ fontSize: 60 }} />}
                    />
                  )}
                </CardContent>
              </Card>

              {mlEvalPlots.length > 0 ? (
                <Grid container spacing={3}>
                  {mlEvalPlots.map((plot) => renderProductCard(plot.name.replace('.png', '').replace(/_/g, ' '), plot.url, plot.name))}
                </Grid>
              ) : null}
            </Box>
          )}

          {/* Phase 5 Tab */}
          {activeTab === 4 && (
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                Deep Learning Model Evaluations
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Confusion matrices and epoch learning curves for raw signals and extracted features.
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <Grid container spacing={3}>
                {phase5Plots.map((plot) => renderProductCard(plot.name, plot.url, plot.url.substring(plot.url.lastIndexOf('/') + 1)))}
              </Grid>
            </Box>
          )}
        </Box>
      )}

      {/* Image Preview Modal with zoom & fullscreen capabilities */}
      <Dialog open={previewOpen} onClose={closePreview} maxWidth="lg" fullWidth>
        <Box sx={{ bgcolor: 'background.paper', position: 'relative' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', px: 3, py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>{previewTitle}</Typography>
            <IconButton onClick={closePreview} color="inherit">
              <CloseIcon />
            </IconButton>
          </Box>

          <DialogContent
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              bgcolor: (theme) => theme.palette.mode === 'dark' ? 'background.default' : 'grey.950',
              p: 2,
              minHeight: '60vh',
              maxHeight: isFullscreen ? '90vh' : '75vh',
              overflow: 'auto',
            }}
          >
            <Box
              sx={{
                transform: `scale(${zoomScale})`,
                transition: 'transform 0.15s ease-out',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                maxWidth: '100%',
              }}
            >
              <img
                src={previewImgUrl}
                alt={previewTitle}
                style={{
                  maxWidth: isFullscreen ? '100vw' : '85%',
                  maxHeight: isFullscreen ? '85vh' : '65vh',
                  objectFit: 'contain',
                }}
              />
            </Box>
          </DialogContent>

          <DialogActions sx={{ justifyContent: 'space-between', px: 3, py: 2, borderTop: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <IconButton onClick={handleZoomIn} color="primary" title="Zoom In">
                <ZoomInIcon />
              </IconButton>
              <IconButton onClick={handleZoomOut} color="primary" title="Zoom Out">
                <ZoomOutIcon />
              </IconButton>
              <IconButton onClick={() => setIsFullscreen(!isFullscreen)} color="primary" title="Toggle Fullscreen">
                <FullscreenIcon />
              </IconButton>
            </Box>
            <Box>
              <Button
                variant="contained"
                startIcon={<DownloadIcon />}
                onClick={() => handleDownload(previewImgUrl, previewTitle.replace(/\s+/g, '_') + '.png')}
              >
                Download File
              </Button>
            </Box>
          </DialogActions>
        </Box>
      </Dialog>
    </Box>
  );
};

export default PipelineResults;
