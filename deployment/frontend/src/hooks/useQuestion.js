/**
 * Custom hook for fetching questions.
 * Prioritizes teacher-created questions from Supabase,
 * falls back to backend (train.csv) questions.
 * Excludes questions the student has already answered.
 */

import { useState, useCallback } from 'react';
import { fetchQuestion, listQuestions } from '../services/api';
import { supabase } from '../services/supabase';
import { useAuth } from '../context/AuthContext';

/**
 * Get list of question IDs the student has already answered.
 * @param {string} studentId - The student's user ID
 * @returns {Promise<Set<string>>} Set of answered question IDs
 */
async function getAnsweredQuestionIds(studentId) {
  if (!studentId) return new Set();

  try {
    const { data, error } = await supabase
      .from('submissions')
      .select('question_id')
      .eq('student_id', studentId);

    if (error) {
      console.error('Error fetching answered questions:', error);
      return new Set();
    }

    return new Set(data.map((s) => s.question_id));
  } catch (err) {
    console.error('Error fetching answered questions:', err);
    return new Set();
  }
}

/**
 * Fetch a random active teacher question from Supabase that the student hasn't answered.
 * @param {Set<string>} answeredIds - Set of already answered question IDs
 * @returns {Promise<Object|null>} Question data or null
 */
async function fetchTeacherQuestion(answeredIds) {
  try {
    // Build query for active questions not yet answered
    let query = supabase
      .from('teacher_questions')
      .select('*')
      .eq('status', 'active');

    // Exclude answered questions
    if (answeredIds.size > 0) {
      const answeredArray = Array.from(answeredIds);
      // Filter to only UUIDs (teacher question IDs have dashes)
      const teacherAnsweredIds = answeredArray.filter((id) => id.includes('-'));
      if (teacherAnsweredIds.length > 0) {
        query = query.not('id', 'in', `(${teacherAnsweredIds.join(',')})`);
      }
    }

    // Get count of available questions
    const { count, error: countError } = await supabase
      .from('teacher_questions')
      .select('*', { count: 'exact', head: true })
      .eq('status', 'active')
      .not('id', 'in', answeredIds.size > 0
        ? `(${Array.from(answeredIds).filter((id) => id.includes('-')).join(',') || 'null'})`
        : '(null)');

    if (countError || !count || count === 0) {
      return null;
    }

    // Get a random offset
    const randomOffset = Math.floor(Math.random() * count);

    // Fetch one random question
    let fetchQuery = supabase
      .from('teacher_questions')
      .select('*')
      .eq('status', 'active');

    // Exclude answered questions
    const teacherAnsweredIds = Array.from(answeredIds).filter((id) => id.includes('-'));
    if (teacherAnsweredIds.length > 0) {
      fetchQuery = fetchQuery.not('id', 'in', `(${teacherAnsweredIds.join(',')})`);
    }

    const { data, error } = await fetchQuery
      .range(randomOffset, randomOffset)
      .single();

    if (error || !data) {
      return null;
    }

    // Transform to match the expected question format
    const options = [];
    if (data.option_a) options.push({ id: 'A', text: data.option_a });
    if (data.option_b) options.push({ id: 'B', text: data.option_b });
    if (data.option_c) options.push({ id: 'C', text: data.option_c });
    if (data.option_d) options.push({ id: 'D', text: data.option_d });

    return {
      id: data.id,
      text: data.question_text,
      options: options,
      has_image: data.has_image || false,
      // Additional metadata for teacher questions
      source: 'teacher',
      teacher_question_id: data.id,
      correct_answer: data.correct_answer,
      expected_misconceptions: data.expected_misconceptions || [],
      category: data.category,
      difficulty: data.difficulty,
    };
  } catch (err) {
    console.error('Error fetching teacher question:', err);
    return null;
  }
}

/**
 * Fetch a random original question from backend that the student hasn't answered.
 * @param {Set<string>} answeredIds - Set of already answered question IDs
 * @returns {Promise<Object|null>} Question data or null
 */
async function fetchOriginalQuestion(answeredIds) {
  try {
    // Get list of all available questions from backend
    const { questions } = await listQuestions();

    if (!questions || questions.length === 0) {
      return null;
    }

    // Filter out answered questions (original IDs are numeric, no dashes)
    const originalAnsweredIds = Array.from(answeredIds).filter((id) => !id.includes('-'));
    const unanswered = questions.filter((q) => !originalAnsweredIds.includes(q.id));

    if (unanswered.length === 0) {
      // All questions answered - return null or allow repeats
      console.log('Student has answered all original questions');
      return null;
    }

    // Pick a random unanswered question
    const randomQuestion = unanswered[Math.floor(Math.random() * unanswered.length)];

    // Fetch the full question data
    const data = await fetchQuestion(randomQuestion.id);
    data.source = 'original';
    return data;
  } catch (err) {
    console.error('Error fetching original question:', err);
    // Fall back to random question from backend
    const data = await fetchQuestion(null);
    data.source = 'original';
    return data;
  }
}

export function useQuestion() {
  const { user } = useAuth();
  const [question, setQuestion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [noMoreQuestions, setNoMoreQuestions] = useState(false);

  const loadQuestion = useCallback(async (questionId = null) => {
    setLoading(true);
    setError(null);
    setNoMoreQuestions(false);

    try {
      // Get list of questions already answered by this student
      const answeredIds = await getAnsweredQuestionIds(user?.id);

      // If requesting a specific question, fetch it directly
      if (questionId) {
        // UUIDs have dashes, original IDs are numeric
        const isUUID = questionId.includes('-');

        if (isUUID) {
          // Fetch specific teacher question
          const { data, error: fetchError } = await supabase
            .from('teacher_questions')
            .select('*')
            .eq('id', questionId)
            .single();

          if (!fetchError && data) {
            const options = [];
            if (data.option_a) options.push({ id: 'A', text: data.option_a });
            if (data.option_b) options.push({ id: 'B', text: data.option_b });
            if (data.option_c) options.push({ id: 'C', text: data.option_c });
            if (data.option_d) options.push({ id: 'D', text: data.option_d });

            const teacherQ = {
              id: data.id,
              text: data.question_text,
              options: options,
              has_image: data.has_image || false,
              source: 'teacher',
              teacher_question_id: data.id,
              correct_answer: data.correct_answer,
              expected_misconceptions: data.expected_misconceptions || [],
              category: data.category,
              difficulty: data.difficulty,
            };
            setQuestion(teacherQ);
            return teacherQ;
          }
        }

        // Fall back to backend for original questions
        const data = await fetchQuestion(questionId);
        data.source = 'original';
        setQuestion(data);
        return data;
      }

      // No specific ID - try teacher question first, then fall back to backend
      // Exclude already answered questions
      const teacherQuestion = await fetchTeacherQuestion(answeredIds);

      if (teacherQuestion) {
        setQuestion(teacherQuestion);
        return teacherQuestion;
      }

      // Fall back to backend (train.csv questions), excluding answered ones
      const originalQuestion = await fetchOriginalQuestion(answeredIds);

      if (originalQuestion) {
        setQuestion(originalQuestion);
        return originalQuestion;
      }

      // No more questions available
      setNoMoreQuestions(true);
      setError('You have answered all available questions! Check back later for new ones.');
      return null;
    } catch (err) {
      setError(err.message || 'Failed to load question');
      return null;
    } finally {
      setLoading(false);
    }
  }, [user?.id]);

  const clearQuestion = useCallback(() => {
    setQuestion(null);
    setError(null);
    setNoMoreQuestions(false);
  }, []);

  return { question, loading, error, loadQuestion, clearQuestion, noMoreQuestions };
}
