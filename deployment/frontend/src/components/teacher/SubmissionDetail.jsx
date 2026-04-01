import React, { useEffect, useRef, useState } from 'react';
import { formatQuestionText, renderMath } from '../../utils/math';
import PredictionRanker from './PredictionRanker';
import FeedbackForm from './FeedbackForm';
import { useAuth } from '../../context/AuthContext';
import { supabase } from '../../services/supabase';

function SubmissionDetail({ submission, onFeedbackSubmitted, onClose }) {
  const { user } = useAuth();
  const [rankings, setRankings] = useState({});
  const [additionalPrediction, setAdditionalPrediction] = useState('');
  const [feedbackText, setFeedbackText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const questionRef = useRef(null);
  const answerRef = useRef(null);
  const explanationRef = useRef(null);

  const existingFeedback = submission.teacher_feedback?.[0];
  const predictions = submission.ai_predictions || [];
  const student = submission.student || submission.profiles;

  useEffect(() => {
    if (questionRef.current) renderMath(questionRef.current);
    if (answerRef.current) renderMath(answerRef.current);
    if (explanationRef.current) renderMath(explanationRef.current);
  }, [submission]);

  const handleSubmitFeedback = async () => {
    if (!feedbackText.trim()) {
      setError('Please provide feedback text for the student');
      return;
    }

    if (Object.keys(rankings).length === 0 && predictions.length > 0) {
      setError('Please rank at least one prediction');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Determine correct prediction label (rank 1)
      const correctLabel = Object.entries(rankings).find(([_, rank]) => rank === 1)?.[0] || null;

      // Insert feedback
      const { data: feedback, error: feedbackError } = await supabase
        .from('teacher_feedback')
        .insert({
          submission_id: submission.id,
          teacher_id: user.id,
          prediction_rankings: rankings,
          additional_prediction: additionalPrediction || null,
          feedback_text: feedbackText,
          correct_prediction_label: correctLabel,
        })
        .select()
        .single();

      if (feedbackError) throw feedbackError;

      // Create notification for student
      const { error: notifError } = await supabase
        .from('notifications')
        .insert({
          user_id: submission.student_id,
          submission_id: submission.id,
          feedback_id: feedback.id,
          title: 'New Teacher Feedback',
          message: `Your teacher reviewed your answer and left feedback.`,
        });

      if (notifError) {
        console.error('Failed to create notification:', notifError);
      }

      onFeedbackSubmitted();
    } catch (err) {
      console.error('Error submitting feedback:', err);
      setError(err.message || 'Failed to submit feedback');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="submission-detail glass-card">
      <div className="detail-header">
        <h2>Submission Review</h2>
        <button className="close-btn" onClick={onClose}>
          <i className="fas fa-times"></i>
        </button>
      </div>

      {/* Student Info */}
      <div className="detail-section">
        <h3>
          <i className="fas fa-user"></i>
          Student
        </h3>
        <p className="student-info">
          {student?.display_name || student?.email?.split('@')[0]}
        </p>
      </div>

      {/* Question */}
      <div className="detail-section">
        <h3>
          <i className="fas fa-question-circle"></i>
          Question
        </h3>
        <div
          ref={questionRef}
          className="question-content"
          dangerouslySetInnerHTML={{ __html: formatQuestionText(submission.question_text) }}
        />
      </div>

      {/* Student's Answer */}
      <div className="detail-section">
        <h3>
          <i className="fas fa-pen"></i>
          Student's Answer
        </h3>
        <div className={`answer-badge ${submission.is_correct ? 'correct' : 'incorrect'}`}>
          <span ref={answerRef} className="answer-option">{submission.selected_answer}</span>
          <span className={`answer-status ${submission.is_correct ? 'correct' : 'incorrect'}`}>
            {submission.is_correct ? 'Correct' : 'Incorrect'}
          </span>
        </div>
      </div>

      {/* Student's Explanation */}
      <div className="detail-section">
        <h3>
          <i className="fas fa-lightbulb"></i>
          Student's Explanation
        </h3>
        <p ref={explanationRef} className="student-explanation">{submission.student_explanation}</p>
      </div>

      {/* AI Predictions */}
      <div className="detail-section">
        <h3>
          <i className="fas fa-robot"></i>
          AI Predictions
        </h3>
        {predictions.length > 0 ? (
          <PredictionRanker
            predictions={predictions}
            rankings={existingFeedback?.prediction_rankings || rankings}
            onChange={setRankings}
            disabled={!!existingFeedback}
          />
        ) : (
          <p className="no-predictions">No predictions available</p>
        )}
      </div>

      {/* Existing Feedback or Feedback Form */}
      {existingFeedback ? (
        <div className="detail-section existing-feedback">
          <h3>
            <i className="fas fa-comment-dots"></i>
            Your Feedback
          </h3>
          <div className="feedback-content">
            <p>{existingFeedback.feedback_text}</p>
            {existingFeedback.additional_prediction && (
              <p className="additional-pred">
                <strong>Additional prediction:</strong> {existingFeedback.additional_prediction}
              </p>
            )}
          </div>
        </div>
      ) : (
        <div className="detail-section">
          <h3>
            <i className="fas fa-comment-dots"></i>
            Your Feedback
          </h3>
          <FeedbackForm
            feedbackText={feedbackText}
            setFeedbackText={setFeedbackText}
            additionalPrediction={additionalPrediction}
            setAdditionalPrediction={setAdditionalPrediction}
            onSubmit={handleSubmitFeedback}
            loading={loading}
            error={error}
          />
        </div>
      )}
    </div>
  );
}

export default SubmissionDetail;
