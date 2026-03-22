/**
 * Application configuration.
 * Uses environment variables for flexibility.
 */

const config = {
  // API base URL - set via REACT_APP_API_BASE_URL environment variable
  apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',

  // Minimum explanation length
  minExplanationLength: 20,

  // Local storage keys
  storageKeys: {
    theme: 'map-student-theme',
    score: 'map-student-score',
    started: 'map-student-started',
  },
};

export default config;
