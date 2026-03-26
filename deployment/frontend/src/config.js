/**
 * Application configuration.
 * Uses environment variables for flexibility.
 */

// Supabase configuration
export const SUPABASE_URL = process.env.REACT_APP_SUPABASE_URL || '';
export const SUPABASE_ANON_KEY = process.env.REACT_APP_SUPABASE_ANON_KEY || '';

const config = {
  // API base URL - set via REACT_APP_API_BASE_URL environment variable
  // Default to Modal URL for production
  apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'https://mkhanshovon--map-misconception-detector-misconceptionapi.modal.run',

  // Modal-specific endpoint URLs (when using Modal deployment)
  modalEndpoints: {
    health: process.env.REACT_APP_MODAL_HEALTH || 'https://mkhanshovon--map-misconception-detector-misconceptionapi-health.modal.run',
    question: process.env.REACT_APP_MODAL_QUESTION || 'https://mkhanshovon--map-misconception-detector-misconceptionapi-3f9f21.modal.run',
    submit: process.env.REACT_APP_MODAL_SUBMIT || 'https://mkhanshovon--map-misconception-detector-misconceptionapi-submit.modal.run',
    predict: process.env.REACT_APP_MODAL_PREDICT || 'https://mkhanshovon--map-misconception-detector-misconceptionapi-f4f678.modal.run',
    labels: process.env.REACT_APP_MODAL_LABELS || 'https://mkhanshovon--map-misconception-detector-misconceptionapi-labels.modal.run',
  },

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
