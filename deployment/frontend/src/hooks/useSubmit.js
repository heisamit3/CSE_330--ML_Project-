/**
 * Custom hook for submitting answers.
 * Handles both original (train.csv) and teacher-created questions.
 */

import { useState, useCallback } from 'react';
import { submitAnswer, predict } from '../services/api';
import { supabase } from '../services/supabase';
import { useAuth } from '../context/AuthContext';

/**
 * Get the correct answer text from a teacher question.
 */
function getCorrectAnswerText(question) {
  if (!question.correct_answer || !question.options) return null;
  const option = question.options.find((o) => o.id === question.correct_answer);
  return option ? option.text : null;
}

export function useSubmit() {
  const { user } = useAuth();
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submit = useCallback(async (question, selectedAnswer, explanation) => {
    setLoading(true);
    setError(null);

    try {
      if (!user?.id) {
        throw new Error('You must be logged in to submit an answer');
      }

      const isTeacherQuestion = question.source === 'teacher';
      let response;

      if (isTeacherQuestion) {
        // For teacher questions, use the /predict endpoint directly
        // since the question may not be in the backend's QUESTIONS_DB
        const correctAnswerText = getCorrectAnswerText(question);
        const selectedAnswerText = question.options.find((o) => o.id === selectedAnswer)?.text || selectedAnswer;
        const isCorrect = question.correct_answer === selectedAnswer;

        const predictResponse = await predict({
          question_id: question.id,
          question_text: question.text,
          mc_answer: selectedAnswerText,
          student_explanation: explanation,
          is_correct: isCorrect,
        });

        // Extract data from nested predict response structure
        const prediction = predictResponse.prediction || {};
        const feedback = predictResponse.feedback || {};
        const ui = predictResponse.ui || {};

        // Transform predict response to match submitAnswer format
        response = {
          success: true,
          feedback: {
            is_correct: isCorrect,
            selected_answer: selectedAnswerText,
            student_explanation: explanation,
            correct_answer: correctAnswerText || feedback.correct_answer,
            predicted_label: prediction.top1_label,
            predicted_category: prediction.predicted_category,
            predicted_misconception: prediction.predicted_misconception,
            top_predictions: prediction.top_predictions || [],
            confidence: prediction.top_predictions?.[0]?.confidence || 0,
            has_misconception: prediction.predicted_category?.includes('Misconception') &&
              prediction.predicted_misconception !== 'NA',
            misconception_name: prediction.predicted_misconception,
            teacher_feedback: feedback.teacher_feedback,
            misconception_card: feedback.misconception_card,
            result_color: ui.result_color || (isCorrect ? '#4ade80' : '#ff6b6b'),
            result_headline: ui.result_headline || (isCorrect ? 'Correct!' : 'Not quite right'),
            result_message: ui.result_message || (isCorrect
              ? 'Great job! Your answer is correct.'
              : 'Let\'s review your answer.'),
            reasoning_label: ui.reasoning_label || (
              prediction.predicted_category?.includes('Misconception') ? 'Misconception Detected' : 'Review'
            ),
            reasoning_message: ui.reasoning_message || feedback.teacher_feedback,
          },
        };
      } else {
        // For original questions, use the standard /api/submit endpoint
        response = await submitAnswer({
          question_id: question.id,
          selected_answer: selectedAnswer,
          student_explanation: explanation,
        });
      }

      if (response.success) {
        // Ensure student_explanation is in feedback so it can be displayed
        if (response.feedback) {
          response.feedback.student_explanation = explanation;
        }

        // Build submission payload for Supabase
        const submissionPayload = {
          student_id: user.id,
          question_id: question.id,
          question_text: question.text,
          selected_answer: response.feedback?.selected_answer || selectedAnswer,
          student_explanation: explanation,
          is_correct: !!response.feedback?.is_correct,
          predicted_label: response.feedback?.predicted_label || null,
          predicted_category: response.feedback?.predicted_category || null,
          predicted_misconception: response.feedback?.predicted_misconception || null,
          ai_predictions: response.feedback?.top_predictions || [],
          top_misconception: response.feedback?.has_misconception
            ? response.feedback?.misconception_name || null
            : null,
          confidence: Number(response.feedback?.confidence || 0),
          // New fields for teacher questions
          question_source: isTeacherQuestion ? 'teacher' : 'original',
          teacher_question_id: isTeacherQuestion ? question.teacher_question_id : null,
        };

        const { error: saveError } = await supabase
          .from('submissions')
          .insert(submissionPayload);

        if (saveError) {
          console.error('Supabase save error:', saveError);
          // Don't throw - we still want to show the feedback
          // throw new Error(saveError.message || 'Failed to save submission to Supabase');
        }

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
  }, [user]);

  const clearFeedback = useCallback(() => {
    setFeedback(null);
    setError(null);
  }, []);

  return { feedback, loading, error, submit, clearFeedback };
}
