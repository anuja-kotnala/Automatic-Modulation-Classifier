import client from './client';

export interface PlotFile {
  name: string;
  category: 'signal' | 'evaluation';
  url: string;
}

export interface GeneratePlotsRequest {
  file_path?: string;
  iq_data?: number[][];
  modulation_type: string;
  sample_rate: number;
}

export interface GeneratedPlotItem {
  name: string;
  filename: string;
  url: string;
}

export interface GeneratePlotsResponse {
  success: boolean;
  plots: GeneratedPlotItem[];
}

export const plotsApi = {
  getPlots: async (): Promise<PlotFile[]> => {
    const response = await client.get<PlotFile[]>('/plots');
    return response.data;
  },

  generatePlots: async (payload: GeneratePlotsRequest): Promise<GeneratePlotsResponse> => {
    const response = await client.post<GeneratePlotsResponse>('/generate-plots', payload);
    return response.data;
  },
};
