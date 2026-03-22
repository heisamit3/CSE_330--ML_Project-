/**
 * API service for communicating with the backend.
 * Supports both local FastAPI and Modal deployment URL patterns.
 */

import config from '../config';

const API_BASE = config.apiBaseUrl;

// Detect if we're using Modal (URLs end with .modal.run)
const isModal = API_BASE.includes('.modal.run');

/**
 * Build the full URL for an endpoint.
 * Modal uses a different URL pattern: base-{method}.modal.run
 * Local/standard FastAPI uses: base/{endpoint}
 */
function buildUrl(endpoint, method = null) {
  if (isModal && method) {
    // Modal pattern: https://user--app-class-method.modal.run
    // The API_BASE should be like: https://user--map-misconception-detector-misconceptionapi
    return `${API_BASE}-${method}.modal.run`;
  }
  // Standard pattern
  return `${API_BASE}${endpoint}`;
}

/**
 * Fetch a question from the API.
 * @param {string} questionId - Optional question ID
 * @returns {Promise<Object>} Question data
 */
export async function fetchQuestion(questionId = null) {
  let url;
  if (isModal) {
    url = buildUrl(null, 'question');
    if (questionId) {
      url += `?question_id=${questionId}`;
    }
  } else {
    url = questionId
      ? `${API_BASE}/api/question?question_id=${questionId}`
      : `${API_BASE}/api/question`;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch question');
  }
  return response.json();
}

/**
 * Submit an answer to the API.
 * @param {Object} data - Submission data
 * @returns {Promise<Object>} Feedback response
 */
export async function submitAnswer(data) {
  const url = isModal ? buildUrl(null, 'submit') : `${API_BASE}/api/submit`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.error || 'Submission failed');
  }

  return response.json();
}

/**
 * Rate feedback quality.
 * @param {Object} data - Rating data
 * @returns {Promise<Object>} Response
 */
export async function rateFeedback(data) {
  const url = isModal ? buildUrl(null, 'rate') : `${API_BASE}/api/rate`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return response.json();
  } catch (e) {
    // Rating is optional, don't fail if it errors
    console.warn('Rating failed:', e);
    return { success: false };
  }
}

/**
 * Check API health.
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const url = isModal ? buildUrl(null, 'health') : `${API_BASE}/health`;
  const response = await fetch(url);
  return response.json();
}

/**
 * List all questions.
 * @returns {Promise<Object>} Questions list
 */
export async function listQuestions() {
  const url = isModal ? buildUrl(null, 'questions') : `${API_BASE}/api/questions/list`;
  const response = await fetch(url);
  return response.json();
}

/**
 * Run prediction with full model.
 * @param {Object} data - Prediction request data
 * @returns {Promise<Object>} Prediction result
 */
export async function predict(data) {
  const url = isModal ? buildUrl(null, 'predict') : `${API_BASE}/predict`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.error || 'Prediction failed');
  }

  return response.json();
}
