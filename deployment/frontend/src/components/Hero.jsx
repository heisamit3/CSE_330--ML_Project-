import React from 'react';

function Hero({ onStart }) {
  return (
    <section className="hero">
      <div className="hero-content">
        <div className="hero-badge">
          <i className="fas fa-user-graduate"></i> Student-facing practice mode
        </div>
        <h1 className="hero-title">
          Choose. Explain.<br />
          <span className="gradient-text">Learn the right method.</span>
        </h1>
        <p className="hero-desc">
          Read a math question, choose one MCQ option, explain how you solved it, and get a clear verdict with the correct method and any misconception highlighted.
        </p>
        <div className="hero-features">
          <div className="feature-item">
            <div className="feature-icon"><i className="fas fa-list-check"></i></div>
            <span>Pick an MCQ</span>
          </div>
          <div className="feature-item">
            <div className="feature-icon"><i className="fas fa-pen"></i></div>
            <span>Explain Thinking</span>
          </div>
          <div className="feature-item">
            <div className="feature-icon"><i className="fas fa-stamp"></i></div>
            <span>Get Verdicts</span>
          </div>
          <div className="feature-item">
            <div className="feature-icon"><i className="fas fa-lightbulb"></i></div>
            <span>Fix Misconceptions</span>
          </div>
        </div>
        <button className="hero-cta" onClick={onStart}>
          <i className="fas fa-play"></i> Start Practice
        </button>
      </div>
    </section>
  );
}

export default Hero;
