import client from './client';

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: Record<string, boolean>;
}

export const healthApi = {
  getHealth: async (): Promise<HealthResponse> => {
    const response = await client.get<HealthResponse>('/health');
    return response.data;
  },

  getHealthDL: async (): Promise<HealthResponse> => {
    const response = await client.get<HealthResponse>('/health-dl');
    return response.data;
  },
};
