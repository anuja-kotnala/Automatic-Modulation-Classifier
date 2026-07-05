import { healthApi } from '../api/healthApi';
import { predictionApi } from '../api/predictionApi';
import { resultsApi } from '../api/resultsApi';
import { plotsApi } from '../api/plotsApi';

// Re-export type definitions for components
export type { HealthResponse } from '../api/healthApi';
export type {
  ChannelImpairments as ChannelImpairments,
  ChannelImpairments as ChannelImpairmentOverrides, // Support alternate naming
  GenerateSignalRequest,
  GenerateSignalResponse,
  PredictRequest,
  PredictResponse,
  PredictDLRequest,
} from '../api/predictionApi';
export type { EvaluationResults } from '../api/resultsApi';
export type { PlotFile, GeneratePlotsRequest, GeneratedPlotItem, GeneratePlotsResponse } from '../api/plotsApi';

// Facade amcApi object to route component triggers cleanly
export const amcApi = {
  getHealth: healthApi.getHealth,
  getHealthDL: healthApi.getHealthDL,
  getPlots: plotsApi.getPlots,
  getResults: resultsApi.getResults,
  getFeatureRankings: resultsApi.getFeatureRankings,
  predict: predictionApi.predict,
  predictDL: predictionApi.predictDL,
  generateSignal: predictionApi.generateSignal,
  generatePlots: plotsApi.generatePlots,
};
