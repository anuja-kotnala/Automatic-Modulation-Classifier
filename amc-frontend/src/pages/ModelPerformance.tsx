import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tab,
  Tabs,
} from '@mui/material';
import {
  BarChart as ChartIcon,
  TableChart as TableIcon,
  Timeline as TimelineIcon,
  BubbleChart as ConfusionIcon,
} from '@mui/icons-material';
import { amcApi } from '../services/amcApi';
import type { PlotFile, EvaluationResults } from '../services/amcApi';
import { API_BASE_URL } from '../api/client';
import AcademicHeader from '../components/Common/AcademicHeader';
import LoadingSkeleton from '../components/Common/LoadingSkeleton';
import EmptyState from '../components/Common/EmptyState';

export const ModelPerformance: React.FC = () => {
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
  const [results, setResults] = useState<EvaluationResults | null>(null);
  const [plots, setPlots] = useState<PlotFile[]>([]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [resData, plotList] = await Promise.all([
        amcApi.getResults(),
        amcApi.getPlots(),
      ]);
      setResults(resData);
      setPlots(plotList);
    } catch (err: any) {
      setError('Failed to fetch model evaluation metrics or plots. Ensure backend evaluations are completed.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Filter evaluation charts
  const mlEvalPlots = plots.filter((p) => p.category === 'evaluation');

  // Compute best and average metrics for academic cards
  let maxAcc = 0;
  let avgF1 = 0;
  let avgPrecision = 0;
  let avgRecall = 0;

  if (results?.summary && results.summary.length > 0) {
    let totalAcc = 0;
    let totalF1 = 0;
    let totalPrec = 0;
    let totalRec = 0;

    results.summary.forEach((row) => {
      // Safely check for headers in different cases
      const getVal = (names: string[]) => {
        const foundKey = Object.keys(row).find((k) => names.includes(k.toLowerCase()));
        return foundKey ? parseFloat(row[foundKey]) : 0;
      };

      const acc = getVal(['accuracy', 'overall_accuracy']);
      const f1 = getVal(['f1', 'f1_score', 'f1-score']);
      const prec = getVal(['precision']);
      const rec = getVal(['recall']);

      if (acc > maxAcc) maxAcc = acc;
      totalAcc += acc;
      totalF1 += f1;
      totalPrec += prec;
      totalRec += rec;
    });

    const len = results.summary.length;
    avgF1 = totalF1 / len;
    avgPrecision = totalPrec / len;
    avgRecall = totalRec / len;
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AcademicHeader
        title="Classifier Model Performance"
        subtitle="Academic analysis comparing accuracy, loss, precision, recall, and training validation history curves for machine learning and deep learning algorithms."
      />

      {error && <Alert severity="error" sx={{ mb: 4 }}>{error}</Alert>}

      {loading ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <LoadingSkeleton variant="stats" count={4} />
          <LoadingSkeleton variant="table" count={5} />
        </Box>
      ) : (
        <Box>
          {/* Dynamic Academic Metrics Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', borderLeft: '5px solid', borderLeftColor: 'primary.main', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 0.5 }}>
                  Peak Validation Accuracy
                </Typography>
                <Typography variant="h3" color="primary" sx={{ fontWeight: 900, mt: 1 }}>
                  {maxAcc > 0 ? `${(maxAcc * 100).toFixed(1)}%` : 'N/A'}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  Highest classification metric
                </Typography>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', borderLeft: '5px solid', borderLeftColor: 'success.main', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 0.5 }}>
                  Mean Macro Precision
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 900, mt: 1, color: 'success.main' }}>
                  {avgPrecision > 0 ? `${(avgPrecision * 100).toFixed(1)}%` : 'N/A'}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  Average positive predictive value
                </Typography>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', borderLeft: '5px solid', borderLeftColor: 'warning.main', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 0.5 }}>
                  Mean Macro Recall
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 900, mt: 1, color: 'warning.main' }}>
                  {avgRecall > 0 ? `${(avgRecall * 100).toFixed(1)}%` : 'N/A'}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  Average sensitivity index
                </Typography>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Paper variant="outlined" sx={{ p: 2.5, textAlign: 'center', borderLeft: '5px solid', borderLeftColor: 'secondary.main', height: '100%' }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 0.5 }}>
                  Mean F1-Harmonic Score
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 900, mt: 1, color: 'secondary.main' }}>
                  {avgF1 > 0 ? `${(avgF1 * 100).toFixed(1)}%` : 'N/A'}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  Harmonic mean of precision & recall
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          {/* Mode Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Tabs value={activeTab} onChange={handleTabChange}>
              <Tab icon={<ChartIcon />} label="Classifier Comparison" iconPosition="start" />
              <Tab icon={<TimelineIcon />} label="Deep Learning History" iconPosition="start" />
              <Tab icon={<ConfusionIcon />} label="Confusion Matrices" iconPosition="start" />
            </Tabs>
          </Box>

          {/* Tab 1: Classifier Comparison */}
          {activeTab === 0 && (
            <Grid container spacing={3}>
              {/* Accuracies bar chart */}
              <Grid size={{ xs: 12, lg: 6 }}>
                <Card sx={{ height: '100%' }}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>
                      Algorithm Accuracy Benchmark
                    </Typography>
                    {results?.summary && results.summary.length > 0 ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                        {results.summary.map((row, index) => {
                          const modelName = row['model'] || row['Model'] || row['Classifier'] || row['classifier'] || 'Unknown';
                          const foundKey = Object.keys(row).find((k) =>
                            ['accuracy', 'overall_accuracy'].includes(k.toLowerCase())
                          );
                          const accuracyVal = foundKey ? parseFloat(row[foundKey]) : 0;
                          return (
                            <Box key={index}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="subtitle2" sx={{ textTransform: 'capitalize', fontWeight: 600 }}>
                                  {modelName.replace(/([A-Z])/g, ' $1')}
                                </Typography>
                                <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 'bold' }}>
                                  {(accuracyVal * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                              <Box sx={{ width: '100%', bgcolor: 'action.hover', borderRadius: 1.5, height: 10, overflow: 'hidden' }}>
                                <Box
                                  sx={{
                                    height: '100%',
                                    bgcolor: 'primary.main',
                                    borderRadius: 1.5,
                                    width: `${accuracyVal * 100}%`,
                                    transition: 'width 1s ease-in-out',
                                  }}
                                />
                              </Box>
                            </Box>
                          );
                        })}
                      </Box>
                    ) : (
                      <EmptyState
                        title="No comparison data found"
                        description="There are no summary stats compiled from model classifications."
                        icon={<ChartIcon sx={{ fontSize: 60 }} />}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Summary Table Card */}
              <Grid size={{ xs: 12, lg: 6 }}>
                <Card sx={{ height: '100%' }}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TableIcon color="primary" /> Metrics Summary (model_performance_summary.csv)
                    </Typography>

                    {results?.summary && results.summary.length > 0 ? (
                      <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                        <Table size="small">
                          <TableHead sx={{ bgcolor: 'background.default' }}>
                            <TableRow>
                              {Object.keys(results.summary[0])
                                .filter((key) => key !== 'model' && key !== 'Classifier' && key !== 'snr_accuracy' && typeof results.summary[0][key] !== 'object')
                                .map((key) => (
                                  <TableCell key={key} sx={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                                    {key.replace('_', ' ')}
                                  </TableCell>
                                ))}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {results.summary.map((row, rIndex) => (
                              <TableRow key={rIndex} hover>
                                {Object.entries(row)
                                  .filter(([key, val]) => key !== 'model' && key !== 'Classifier' && key !== 'snr_accuracy' && typeof val !== 'object')
                                  .map(([key, val]: any) => (
                                    <TableCell key={key}>
                                      {typeof val === 'number'
                                        ? val.toFixed(4)
                                        : typeof val === 'string' && !isNaN(Number(val))
                                        ? parseFloat(val).toFixed(4)
                                        : val}
                                    </TableCell>
                                  ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    ) : (
                      <EmptyState
                        title="Summary Table Unavailable"
                        description="No models have completed the baseline evaluation metrics compilation pipeline."
                        icon={<TableIcon sx={{ fontSize: 60 }} />}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Tab 2: Deep Learning History */}
          {activeTab === 1 && (
            <Grid container spacing={3}>
              {[
                { name: 'CNN1D (Raw IQ Waves) Learning History', url: '/static/dl_eval/cnn1d_raw_learning_curves.png' },
                { name: 'CNN-LSTM (Raw IQ Waves) Learning History', url: '/static/dl_eval/cnnlstm_raw_learning_curves.png' },
                { name: 'CNN2D (Spectrogram) Learning History', url: '/static/dl_eval/cnn2d_raw_learning_curves.png' }
              ].map((item) => (
                <Grid size={{ xs: 12, md: 6 }} key={item.name}>
                  <Card sx={{ transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                        {item.name}
                      </Typography>
                      <Box sx={{ p: 1.5, bgcolor: 'background.default', display: 'flex', justifyContent: 'center', borderRadius: 2 }}>
                        <img
                          src={`${API_BASE_URL}${item.url}`}
                          alt={item.name}
                          style={{ maxWidth: '100%', height: 'auto', borderRadius: 4 }}
                          onError={(e) => {
                            (e.target as any).style.display = 'none';
                          }}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}

          {/* Tab 3: Confusion Matrices */}
          {activeTab === 2 && (
            <Grid container spacing={3}>
              {/* Dynamic Confusions from ML */}
              {mlEvalPlots.map((plot) => (
                <Grid size={{ xs: 12, md: 6 }} key={plot.name}>
                  <Card sx={{ transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2, textTransform: 'capitalize' }}>
                        {plot.name.replace('.png', '').replace(/_/g, ' ')}
                      </Typography>
                      <Box sx={{ p: 1.5, bgcolor: (theme) => theme.palette.mode === 'dark' ? 'background.default' : 'common.white', display: 'flex', justifyContent: 'center', borderRadius: 2 }}>
                        <img
                          src={`${API_BASE_URL}${plot.url}`}
                          alt={plot.name}
                          style={{ maxWidth: '100%', height: 'auto', borderRadius: 4 }}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}

              {/* DL Confusions */}
              {[
                { name: 'CNN1D (Raw IQ) Confusion Matrix', url: '/static/dl_eval/cnn1d_raw_confusion_matrix.png' },
                { name: 'CNN-LSTM Confusion Matrix', url: '/static/dl_eval/cnnlstm_raw_confusion_matrix.png' },
                { name: 'CNN2D Spectrogram Confusion Matrix', url: '/static/dl_eval/cnn2d_raw_confusion_matrix.png' }
              ].map((item) => (
                <Grid size={{ xs: 12, md: 6 }} key={item.name}>
                  <Card sx={{ transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                        {item.name}
                      </Typography>
                      <Box sx={{ p: 1.5, bgcolor: (theme) => theme.palette.mode === 'dark' ? 'background.default' : 'common.white', display: 'flex', justifyContent: 'center', borderRadius: 2 }}>
                        <img
                          src={`${API_BASE_URL}${item.url}`}
                          alt={item.name}
                          style={{ maxWidth: '100%', height: 'auto', borderRadius: 4 }}
                          onError={(e) => {
                            (e.target as any).style.display = 'none';
                          }}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ModelPerformance;
