import axios, { AxiosError } from 'axios';
import type { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

interface RetryConfig extends InternalAxiosRequestConfig {
  _retryCount?: number;
}

const client: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Configure automatic retry logic
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1500;

client.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const config = error.config as RetryConfig;

    // Retry only on network/timeout or 503/504 errors
    const shouldRetry =
      config &&
      (!error.response || error.response.status === 503 || error.response.status === 504 || error.code === 'ECONNABORTED');

    if (shouldRetry) {
      config._retryCount = config._retryCount ?? 0;

      if (config._retryCount < MAX_RETRIES) {
        config._retryCount += 1;
        // Wait before retrying
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS * config._retryCount!));
        return client(config);
      }
    }

    // Format error response before throwing
    const message = (error.response?.data as any)?.detail || error.message || 'An unexpected error occurred.';
    return Promise.reject({
      message,
      status: error.response?.status,
      originalError: error,
    });
  }
);

export default client;
