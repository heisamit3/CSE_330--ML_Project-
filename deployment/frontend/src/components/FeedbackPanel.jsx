import React, { useState, useEffect, useRef } from 'react';
import { renderMath } from '../utils/math';
import { rateFeedback } from '../services/api';

function FeedbackPanel({ feedback, questionId, onNext }) {
  const [rating, setRating] = useState(0);
  const [rated, setRated] = useState(false);

  const solutionRef = useRef(null);
  const yourAnswerRef = useRef(null);
  const correctAnswerRef = useRef(null);

  useEffect(() => {
    // Render math in feedback sections
    if (solutionRef.current) renderMath(solutionRef.current);
    if (yourAnswerRef.current) renderMath(yourAnswerRef.current);
    if (correctAnswerRef.current) renderMath(correctAnswerRef.current);
  }, [feedback]);

  const handleRate = async (value) => {
    setRating(value);
    try {
      await rateFeedback({ question_id: questionId, rating: value });
      setTimeout(() => setRated(true), 400);
    } catch (e) {
      console.warn('Could not send rating:', e);
    }
  };

  if (!feedback) return null;

  return (
    <div className="feedback-panel">
      {/* Result Banner */}
      <div className="glass-card result-banner">
        <div className={`result-icon-wrap ${feedback.is_correct ? 'correct' : 'incorrect'}`}>
          <i className={`fas fa-${feedback.is_correct ? 'check' : 'times'}-circle`}></i>
        </div>
        <div className="result-text">
          <h2 style={{ color: feedback.result_color }}>{feedback.result_headline}</h2>
          <p>{feedback.result_message}</p>
          <div className="result-meta">
            <span className={`meta-pill ${feedback.is_correct ? 'meta-pill-correct' : 'meta-pill-incorrect'}`}>
              {feedback.answer_label}
            </span>
            <span className={`meta-pill ${feedback.has_misconception ? 'meta-pill-warning' : 'meta-pill-neutral'}`}>
              {feedback.reasoning_label}
            </span>
          </div>
        </div>
      </div>

      {/* Choice Comparison */}
      <div className="glass-card comparison-card">
        <h3 className="card-title">
          <i className="fas fa-arrows-left-right"></i> Choice Review
        </h3>
        <div className="comparison-grid">
          <div className="comparison-item your-answer">
            <div className="comparison-label">Your Choice</div>
            <div className="comparison-value" ref={yourAnswerRef}>
              {feedback.selected_answer}
            </div>
          </div>
          <div className="comparison-vs">
            <i className="fas fa-arrows-left-right"></i>
          </div>
          <div className="comparison-item correct-answer">
            <div className="comparison-label">Correct Answer</div>
            <div className="comparison-value" ref={correctAnswerRef}>
              {feedback.correct_answer}
            </div>
          </div>
        </div>
      </div>

      {/* Your Explanation */}
      {feedback.student_explanation && (
        <div className="glass-card explanation-card">
          <h3 className="card-title">
            <i className="fas fa-pen"></i> Your Explanation
          </h3>
          <div className="explanation-content">
            {feedback.student_explanation}
          </div>
        </div>
      )}

      {/* Top Predictions (NEW - from real model) */}
      {feedback.top_predictions && feedback.top_predictions.length > 0 && (
        <div className="glass-card predictions-card">
          <h3 className="card-title">
            <i className="fas fa-chart-bar"></i> Model Predictions
          </h3>
          <div className="predictions-list">
            {feedback.top_predictions.slice(0, 3).map((pred, idx) => (
              <div key={idx} className="prediction-item">
                <div className="prediction-label">{pred.label}</div>
                <div className="prediction-bar-container">
                  <div
                    className="prediction-bar"
                    style={{ width: `${Math.min(pred.confidence * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Solution/Teacher Feedback */}
      <div className="glass-card solution-card">
        <h3 className="card-title">
          <i className="fas fa-lightbulb"></i> Teacher Feedback
        </h3>
        <div className="solution-content" ref={solutionRef}>
          {feedback.teacher_feedback || feedback.correct_explanation}
        </div>
      </div>

      {/* Misconception Card (NEW - from KB) */}
      {feedback.has_misconception && feedback.misconception_card && (
        <div className="glass-card misconception-card kb-card">
          <div className="misconception-header">
            <div className="misconception-icon-wrap">
              <i className="fas fa-book-open"></i>
            </div>
            <div>
              <h3>Knowledge Card: {feedback.misconception_name}</h3>
              <p className="misconception-subtitle">Understanding this misconception</p>
            </div>
          </div>
          <div className="kb-content">
            {feedback.misconception_card.plain_definition && (
              <div className="kb-section">
                <h4><i className="fas fa-info-circle"></i> Definition</h4>
                <p>{feedback.misconception_card.plain_definition}</p>
              </div>
            )}
            {feedback.misconception_card.typical_wrong_thought && (
              <div className="kb-section">
                <h4><i className="fas fa-brain"></i> Common Wrong Thoughts</h4>
                <ul>
                  {feedback.misconception_card.typical_wrong_thought.map((thought, i) => (
                    <li key={i}>{thought}</li>
                  ))}
                </ul>
              </div>
            )}
            {feedback.misconception_card.fix_steps && (
              <div className="kb-section">
                <h4><i className="fas fa-tools"></i> How to Fix It</h4>
                <ol>
                  {feedback.misconception_card.fix_steps.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Simple Misconception (fallback without KB card) */}
      {feedback.has_misconception && !feedback.misconception_card && (
        <div className="glass-card misconception-card">
          <div className="misconception-header">
            <div className="misconception-icon-wrap">
              <i className="fas fa-exclamation-triangle"></i>
            </div>
            <div>
              <h3>{feedback.misconception_name}</h3>
              <p className="misconception-subtitle">This is the idea we should correct next</p>
            </div>
          </div>
          <div className="misconception-body">
            {feedback.what_went_wrong && (
              <div className="misconception-section">
                <h4><i className="fas fa-search"></i> What went wrong</h4>
                <p>{feedback.what_went_wrong}</p>
              </div>
            )}
            <div className="misconception-section tip-section">
              <h4><i className="fas fa-lightbulb"></i> How to fix it</h4>
              <p>{feedback.misconception_tip}</p>
            </div>
          </div>
        </div>
      )}

      {/* Rating */}
      <div className="glass-card rating-card">
        <h3 className="card-title">
          <i className="fas fa-star"></i> Was this feedback helpful?
        </h3>
        {!rated ? (
          <div className="star-rating">
            {[1, 2, 3, 4, 5].map((value) => (
              <button
                key={value}
                className={`star-btn ${rating >= value ? 'active' : ''}`}
                onClick={() => handleRate(value)}
              >
                <i className="fas fa-star"></i>
              </button>
            ))}
          </div>
        ) : (
          <div className="rating-thanks">
            <i className="fas fa-heart"></i> Thanks for your feedback!
          </div>
        )}
      </div>

      {/* Next Button */}
      <button className="btn btn-primary btn-full btn-next" onClick={onNext}>
        <i className="fas fa-arrow-right"></i> Next Question
      </button>
    </div>
  );
}

export default FeedbackPanel;
