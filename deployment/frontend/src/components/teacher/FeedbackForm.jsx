import React from 'react';

// Common misconception labels for the dropdown
const COMMON_MISCONCEPTIONS = [
  'Additive',
  'Incomplete',
  'Wrong_fraction',
  'Procedural',
  'Sign_error',
  'Order_of_operations',
  'Distributive_property',
  'Fraction_misconception',
  'Decimal_misconception',
  'Percentage_error',
  'Unit_confusion',
  'Rounding_error',
  'Other',
];

function FeedbackForm({
  feedbackText,
  setFeedbackText,
  additionalPrediction,
  setAdditionalPrediction,
  onSubmit,
  loading,
  error,
}) {
  return (
    <div className="feedback-form">
      {error && (
        <div className="form-error">
          <i className="fas fa-exclamation-circle"></i>
          {error}
        </div>
      )}

      <div className="form-group">
        <label htmlFor="feedbackText">Feedback for Student</label>
        <textarea
          id="feedbackText"
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          placeholder="Provide constructive feedback to help the student understand their misconception..."
          rows={4}
          disabled={loading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="additionalPrediction">
          Additional Prediction (Optional)
          <span className="label-hint">If the AI missed the correct misconception</span>
        </label>
        <select
          id="additionalPrediction"
          value={additionalPrediction}
          onChange={(e) => setAdditionalPrediction(e.target.value)}
          disabled={loading}
        >
          <option value="">-- Select if needed --</option>
          {COMMON_MISCONCEPTIONS.map((m) => (
            <option key={m} value={m}>
              {m.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      <button
        className="submit-feedback-btn"
        onClick={onSubmit}
        disabled={loading || !feedbackText.trim()}
      >
        {loading ? (
          <>
            <i className="fas fa-spinner fa-spin"></i>
            Submitting...
          </>
        ) : (
          <>
            <i className="fas fa-paper-plane"></i>
            Submit Feedback
          </>
        )}
      </button>
    </div>
  );
}

export default FeedbackForm;
