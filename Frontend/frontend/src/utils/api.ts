import axios, { AxiosError } from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add timeout
  timeout: 5000,
  // Allow credentials
  withCredentials: true,
});

// Add request interceptor
api.interceptors.request.use(
  (config) => {
    // Get token from localStorage or auth state
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor with better error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout');
      return Promise.reject(new Error('Request timeout - Please check your connection'));
    }

    if (!error.response) {
      console.error('Network error:', error);
      return Promise.reject(new Error('Network error - Please check if the API server is running'));
    }

    switch (error.response.status) {
      case 401:
        console.error('Unauthorized access');
        // Handle unauthorized access
        break;
      case 403:
        console.error('Forbidden access');
        break;
      case 404:
        console.error('Resource not found');
        break;
      case 500:
        console.error('Server error');
        break;
      default:
        console.error('API error:', error);
    }

    return Promise.reject(error);
  }
);

export default api;