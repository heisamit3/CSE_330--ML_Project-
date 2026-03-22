/**
 * Custom hook for fetching questions.
 */

import { useState, useCallback } from 'react';
import { fetchQuestion } from '../services/api';

export function useQuestion() {
  const [question, setQuestion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadQuestion = useCallback(async (questionId = null) => {
    setLoading(true);
    setError(null);

    try {
      const data = await fetchQuestion(questionId);
      setQuestion(data);
      return data;
    } catch (err) {
      setError(err.message || 'Failed to load question');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearQuestion = useCallback(() => {
    setQuestion(null);
    setError(null);
  }, []);

  return { question, loading, error, loadQuestion, clearQuestion };
}
