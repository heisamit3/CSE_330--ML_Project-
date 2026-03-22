import React, { useEffect, useRef } from 'react';
import { renderMath, formatQuestionText } from '../utils/math';

function QuestionCard({ question, questionNumber, loading, onSkip }) {
  const questionRef = useRef(null);

  useEffect(() => {
    if (questionRef.current && question) {
      renderMath(questionRef.current);
    }
  }, [question]);

  return (
    <div className="glass-card question-card">
      <div className="card-header">
        <div className="question-badge">
          <i className="fas fa-question-circle"></i>
          Question <span>{questionNumber}</span>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={onSkip} title="Skip this question">
          <i className="fas fa-forward"></i> New Question
        </button>
      </div>
      <div
        className="question-text"
        ref={questionRef}
        dangerouslySetInnerHTML={{
          __html: loading
            ? 'Loading question...'
            : question
            ? formatQuestionText(question.text)
            : 'No question loaded',
        }}
      />
    </div>
  );
}

export default QuestionCard;
