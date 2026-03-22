/**
 * Custom hook for managing score state.
 */

import { useState, useEffect, useCallback } from 'react';
import config from '../config';

const STORAGE_KEY = config.storageKeys.score;

export function useScore() {
  const [score, setScore] = useState({ correct: 0, wrong: 0, total: 0 });

  // Load score from sessionStorage on mount
  useEffect(() => {
    const saved = sessionStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        setScore(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to parse saved score', e);
      }
    }
  }, []);

  // Save score to sessionStorage when it changes
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(score));
  }, [score]);

  const addResult = useCallback((isCorrect) => {
    setScore((prev) => ({
      correct: prev.correct + (isCorrect ? 1 : 0),
      wrong: prev.wrong + (isCorrect ? 0 : 1),
      total: prev.total + 1,
    }));
  }, []);

  const resetScore = useCallback(() => {
    setScore({ correct: 0, wrong: 0, total: 0 });
  }, []);

  return { score, addResult, resetScore };
}
