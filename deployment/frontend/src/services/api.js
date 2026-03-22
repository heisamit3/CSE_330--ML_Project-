/**
 * API service for communicating with the backend.
 * Supports both local FastAPI and Modal deployment URL patterns.
 */

import config from '../config';

const API_BASE = config.apiBaseUrl;

// Detect if we're using Modal (URLs end with .modal.run)
const isModal = API_BASE.includes('.modal.run');

/**
 * Get the URL for a Modal endpoint.
 */
function getModalUrl(endpoint) {
  return config.modalEndpoints[endpoint] || null;
}

/**
 * Fetch a question from the API.
 * @param {string} questionId - Optional question ID
 * @returns {Promise<Object>} Question data
 */
export async function fetchQuestion(questionId = null) {
  let url;
  if (isModal) {
    url = getModalUrl('question');
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
  const url = isModal ? getModalUrl('submit') : `${API_BASE}/api/submit`;

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
  const url = isModal ? getModalUrl('rate') : `${API_BASE}/api/rate`;

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
  const url = isModal ? getModalUrl('health') : `${API_BASE}/health`;
  const response = await fetch(url);
  return response.json();
}

/**
 * List all questions.
 * @returns {Promise<Object>} Questions list
 */
export async function listQuestions() {
  const url = isModal ? getModalUrl('questions') : `${API_BASE}/api/questions/list`;
  const response = await fetch(url);
  return response.json();
}

/**
 * Run prediction with full model.
 * @param {Object} data - Prediction request data
 * @returns {Promise<Object>} Prediction result
 */
export async function predict(data) {
  const url = isModal ? getModalUrl('predict') : `${API_BASE}/predict`;

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
