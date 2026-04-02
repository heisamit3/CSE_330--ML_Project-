import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useScore } from '../hooks/useScore';
import NotificationBell from './NotificationBell';
import BrandMark from './BrandMark';

function Navbar({ theme, onToggleTheme, user, profile, onSignOut }) {
  const navigate = useNavigate();
  const { score } = useScore();

  const handleSignOut = async () => {
    console.log('Sign out clicked!');
    try {
      if (onSignOut) {
        console.log('Calling signOut...');
        await onSignOut();
        console.log('SignOut complete');
      }
      navigate('/login');
    } catch (err) {
      console.error('Sign out error:', err);
      navigate('/login');
    }
  };

  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          <div className="brand-icon">
            <BrandMark />
          </div>
          <div className="brand-text">
            <span className="brand-name">MAP</span>
            <span className="brand-sub">{profile?.role === 'teacher' ? 'Teacher' : 'Student'} Practice</span>
          </div>
        </Link>

        {/* Score display - only for students in practice */}
        {user && profile?.role === 'student' && (
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
        )}

        {/* Teacher navigation */}
        {user && profile?.role === 'teacher' && (
          <div className="nav-center">
            <div className="nav-links">
              <Link to="/dashboard" className="nav-link">
                <i className="fas fa-th-large"></i>
                Dashboard
              </Link>
            </div>
          </div>
        )}

        <div className="nav-actions">
          <button className="theme-toggle" onClick={onToggleTheme} title="Toggle theme">
            <i className={`fas fa-${theme === 'dark' ? 'moon' : 'sun'}`}></i>
          </button>

          {user ? (
            <div className="user-menu">
              {profile?.role === 'student' && <NotificationBell />}
              <span className="user-name">
                {profile?.display_name || user.email?.split('@')[0]}
              </span>
              <button className="sign-out-btn" onClick={handleSignOut} title="Sign out">
                <i className="fas fa-sign-out-alt"></i>
              </button>
            </div>
          ) : (
            <div className="auth-buttons">
              <Link to="/login" className="nav-btn nav-btn-secondary">
                Sign In
              </Link>
              <Link to="/signup" className="nav-btn nav-btn-primary">
                Sign Up
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
