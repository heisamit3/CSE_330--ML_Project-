import React from 'react';

function SubmissionList({ submissions, loading, onSelect, selected }) {
  if (loading) {
    return (
      <div className="submission-list loading">
        <div className="loading-spinner"></div>
        <p>Loading submissions...</p>
      </div>
    );
  }

  if (submissions.length === 0) {
    return (
      <div className="submission-list empty">
        <i className="fas fa-inbox"></i>
        <p>No submissions found</p>
      </div>
    );
  }

  const truncateText = (text, maxLength = 100) => {
    if (!text) return '';
    // Remove LaTeX and clean up
    const cleaned = text.replace(/\\\(.*?\\\)/g, '[math]').replace(/\\\[.*?\\\]/g, '[math]');
    return cleaned.length > maxLength ? cleaned.substring(0, maxLength) + '...' : cleaned;
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="submission-list">
      {submissions.map((submission) => {
        const hasReview = submission.teacher_feedback?.length > 0;
        const isSelected = selected?.id === submission.id;
        const student = submission.student || submission.profiles;

        return (
          <div
            key={submission.id}
            className={`submission-item ${isSelected ? 'selected' : ''} ${hasReview ? 'reviewed' : 'pending'}`}
            onClick={() => onSelect(submission)}
          >
            <div className="submission-header">
              <span className="student-name">
                <i className="fas fa-user"></i>
                {student?.display_name || student?.email?.split('@')[0] || 'Unknown'}
              </span>
              <span className={`submission-status ${hasReview ? 'reviewed' : 'pending'}`}>
                {hasReview ? (
                  <>
                    <i className="fas fa-check-circle"></i>
                    Reviewed
                  </>
                ) : (
                  <>
                    <i className="fas fa-clock"></i>
                    Pending
                  </>
                )}
              </span>
            </div>

            <div className="submission-preview">
              <p className="question-preview">{truncateText(submission.question_text)}</p>
            </div>

            <div className="submission-meta">
              <span className={`answer-status ${submission.is_correct ? 'correct' : 'incorrect'}`}>
                <i className={`fas fa-${submission.is_correct ? 'check' : 'times'}`}></i>
                {submission.is_correct ? 'Correct' : 'Incorrect'}
              </span>
              <span className="submission-date">
                {formatDate(submission.created_at)}
              </span>
            </div>

            {submission.top_misconception && (
              <div className="submission-misconception">
                <i className="fas fa-exclamation-triangle"></i>
                {submission.top_misconception.split(':').pop()}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default SubmissionList;
