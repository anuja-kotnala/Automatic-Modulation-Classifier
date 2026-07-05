import client from './client';

export interface ChannelImpairments {
  awgn_enabled?: boolean;
  rayleigh_enabled?: boolean;
  rician_enabled?: boolean;
  frequency_offset_enabled?: boolean;
  phase_noise_enabled?: boolean;
  iq_imbalance_enabled?: boolean;
  timing_offset_enabled?: boolean;
  multipath_enabled?: boolean;
  clock_drift_enabled?: boolean;

  awgn_snr_db?: number;
  rayleigh_doppler_freq?: number;
  rician_k_factor?: number;
  rician_doppler_freq?: number;
  frequency_offset_hz?: number;
  phase_noise_std_dev?: number;
  iq_imbalance_amplitude_db?: number;
  iq_imbalance_phase_deg?: number;
  timing_offset_fractional_delay?: number;
  multipath_delays?: number[];
  multipath_gains_db?: number[];
  clock_drift_ppm?: number;
}

export interface GenerateSignalRequest {
  modulation_type: string;
  snr_db: number;
  sample_rate: number;
  num_samples: number;
  impairments?: ChannelImpairments;
}

export interface GenerateSignalResponse {
  success: boolean;
  modulation_type: string;
  snr_db: number;
  iq_data: number[][];
  file_path?: string;
}

export interface PredictRequest {
  file_path?: string;
  iq_data?: number[][];
  model_name: string;
}

export interface PredictResponse {
  success: boolean;
  predicted_class: string;
  confidence: number;
  probabilities?: Record<string, number>;
  features: Record<string, number>;
}

export interface PredictDLRequest {
  file_path?: string;
  iq_data?: number[][];
  model_name: 'cnn1d' | 'cnnlstm' | 'cnn2d' | string;
}

export const predictionApi = {
  generateSignal: async (payload: GenerateSignalRequest): Promise<GenerateSignalResponse> => {
    const response = await client.post<GenerateSignalResponse>('/generate-signal', payload);
    return response.data;
  },

  predict: async (payload: PredictRequest): Promise<PredictResponse> => {
    const response = await client.post<PredictResponse>('/predict', payload);
    return response.data;
  },

  predictDL: async (payload: PredictDLRequest): Promise<PredictResponse> => {
    const response = await client.post<PredictResponse>('/predict-dl', payload);
    return response.data;
  },
};
