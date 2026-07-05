import client from './client';

export interface EvaluationResults {
  summary: Array<Record<string, any>>;
  available_files: string[];
}

export const resultsApi = {
  getResults: async (): Promise<EvaluationResults> => {
    const response = await client.get<EvaluationResults>('/results');
    return response.data;
  },

  getFeatureRankings: async (): Promise<Array<Record<string, string>>> => {
    try {
      const response = await client.get<string>('/static/analysis/feature_rankings.csv', {
        responseType: 'text',
      });
      
      const csvText = response.data;
      const lines = csvText.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
      if (lines.length === 0) return [];
      
      const headers = lines[0].split(',');
      return lines.slice(1).map((line) => {
        const values = line.split(',');
        const obj: Record<string, string> = {};
        headers.forEach((header, idx) => {
          obj[header] = values[idx] || '';
        });
        return obj;
      });
    } catch (error) {
      console.warn('Feature rankings CSV not available on disk yet.', error);
      return [];
    }
  },
};
