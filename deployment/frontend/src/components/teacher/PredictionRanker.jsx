import React from 'react';

function PredictionRanker({ predictions, rankings, onChange, disabled }) {
  const parsePredictionLabel = (label) => {
    const [family, ...rest] = String(label || '').split(':');
    return {
      family: family || label,
      detail: rest.join(':'),
    };
  };

  const handleRankChange = (label, newRank) => {
    if (disabled) return;

    const updated = { ...rankings };

    // Find current holder of the new rank and swap
    const currentHolder = Object.keys(updated).find(k => updated[k] === newRank);
    if (currentHolder && currentHolder !== label) {
      updated[currentHolder] = rankings[label] || 0;
    }
    updated[label] = newRank;

    // Remove any 0 rankings
    Object.keys(updated).forEach(k => {
      if (updated[k] === 0) delete updated[k];
    });

    onChange(updated);
  };

  const getRankClass = (rank) => {
    if (rank === 1) return 'rank-1';
    if (rank === 2) return 'rank-2';
    if (rank === 3) return 'rank-3';
    return '';
  };

  return (
    <div className="prediction-ranker">
      <p className="ranker-instructions">
        {disabled
          ? 'Predictions have been ranked.'
          : 'Rank the predictions (1 = most accurate):'}
      </p>
      <div className="predictions-list">
        {predictions.map((pred, index) => {
          const currentRank = rankings[pred.label] || 0;
          const parsedLabel = parsePredictionLabel(pred.label);

          return (
            <div key={pred.label} className="prediction-rank-item">
              <div className="prediction-info">
                <div className="prediction-label-wrap">
                  <span className="prediction-family">{parsedLabel.family}</span>
                  {parsedLabel.detail && (
                    <span className="prediction-label">
                      {parsedLabel.detail}
                    </span>
                  )}
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${(pred.confidence * 100).toFixed(0)}%` }}
                  />
                  <span className="confidence-text">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="rank-selector">
                {[1, 2, 3].map(rank => (
                  <button
                    key={rank}
                    className={`rank-btn ${currentRank === rank ? 'active ' + getRankClass(rank) : ''}`}
                    onClick={() => handleRankChange(pred.label, rank)}
                    disabled={disabled}
                    title={`Rank as ${rank}`}
                  >
                    {rank}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default PredictionRanker;
