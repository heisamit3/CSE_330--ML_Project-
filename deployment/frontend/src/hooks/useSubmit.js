/**
 * Custom hook for submitting answers.
 */

import { useState, useCallback } from 'react';
import { submitAnswer } from '../services/api';

export function useSubmit() {
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submit = useCallback(async (questionId, selectedAnswer, explanation) => {
    setLoading(true);
    setError(null);

    try {
      const response = await submitAnswer({
        question_id: questionId,
        selected_answer: selectedAnswer,
        student_explanation: explanation,
      });

      if (response.success) {
        setFeedback(response.feedback);
        return response.feedback;
      } else {
        throw new Error(response.error || 'Submission failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to submit answer');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearFeedback = useCallback(() => {
    setFeedback(null);
    setError(null);
  }, []);

  return { feedback, loading, error, submit, clearFeedback };
}
