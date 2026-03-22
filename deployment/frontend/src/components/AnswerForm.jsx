import React, { useState, useEffect, useRef } from 'react';
import { renderMath } from '../utils/math';

function AnswerForm({ question, loading, onSubmit }) {
  const [selectedAnswer, setSelectedAnswer] = useState('');
  const [explanation, setExplanation] = useState('');
  const optionsRef = useRef(null);

  // Reset form when question changes
  useEffect(() => {
    setSelectedAnswer('');
    setExplanation('');
  }, [question?.id]);

  // Render math in options
  useEffect(() => {
    if (optionsRef.current && question) {
      renderMath(optionsRef.current);
    }
  }, [question]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(selectedAnswer, explanation);
  };

  if (!question) return null;

  return (
    <div className="glass-card answer-card">
      <div className="card-header">
        <h3><i className="fas fa-pencil-ruler"></i> Your Response</h3>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">
            <i className="fas fa-list-ul"></i> Step 1: Choose one answer
          </label>
          <div className="option-grid" ref={optionsRef}>
            {question.options.map((option) => (
              <label
                key={option.id}
                className={`option-card ${selectedAnswer === option.text ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="selectedAnswer"
                  value={option.text}
                  checked={selectedAnswer === option.text}
                  onChange={(e) => setSelectedAnswer(e.target.value)}
                  className="option-input"
                />
                <span className="option-badge">{option.id}</span>
                <div className="option-content">
                  <div className="option-title">Option {option.id}</div>
                  <div className="option-text">{option.text}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">
            <i className="fas fa-comment-dots"></i> Step 2: Explain how you solved it
            <span className="label-hint">Use a short step-by-step explanation</span>
          </label>
          <textarea
            className="form-textarea"
            rows="5"
            placeholder="Example: I found one fifth first, then multiplied by 3 because the question asks for three fifths."
            value={explanation}
            onChange={(e) => setExplanation(e.target.value)}
            required
          />
          <div className="char-count">
            <span>{explanation.length}</span> characters
            <span className="char-hint">(aim for at least 20)</span>
          </div>
        </div>

        <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
          {loading ? (
            <>
              <div className="btn-loader">
                <div className="spinner"></div>
              </div>
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <i className="fas fa-paper-plane"></i>
              <span>Check My Answer</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}

export default AnswerForm;
