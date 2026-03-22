import React from 'react';

function Navbar({ score, theme, onToggleTheme }) {
  return (
    <nav className="navbar">
      <div className="nav-container">
        <a href="/" className="nav-brand">
          <div className="brand-icon">
            <i className="fas fa-graduation-cap"></i>
          </div>
          <div className="brand-text">
            <span className="brand-name">MAP</span>
            <span className="brand-sub">Student Practice</span>
          </div>
        </a>
        <div className="nav-center">
          <div className="score-display">
            <div className="score-item">
              <i className="fas fa-check-circle" style={{ color: 'var(--success)' }}></i>
              <span>{score.correct}</span>
            </div>
            <div className="score-divider"></div>
            <div className="score-item">
              <i className="fas fa-times-circle" style={{ color: 'var(--danger)' }}></i>
              <span>{score.wrong}</span>
            </div>
            <div className="score-divider"></div>
            <div className="score-item">
              <i className="fas fa-layer-group" style={{ color: 'var(--accent-primary)' }}></i>
              <span>{score.total}</span>
            </div>
          </div>
        </div>
        <div className="nav-actions">
          <button className="theme-toggle" onClick={onToggleTheme} title="Toggle theme">
            <i className={`fas fa-${theme === 'dark' ? 'moon' : 'sun'}`}></i>
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
