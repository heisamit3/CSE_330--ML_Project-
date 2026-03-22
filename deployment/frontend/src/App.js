import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import QuestionCard from './components/QuestionCard';
import AnswerForm from './components/AnswerForm';
import FeedbackPanel from './components/FeedbackPanel';
import Toast from './components/Toast';
import { useScore } from './hooks/useScore';
import { useQuestion } from './hooks/useQuestion';
import { useSubmit } from './hooks/useSubmit';
import config from './config';

function App() {
  const [theme, setTheme] = useState('dark');
  const [started, setStarted] = useState(false);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [toasts, setToasts] = useState([]);

  const { score, addResult } = useScore();
  const { question, loading: questionLoading, error: questionError, loadQuestion } = useQuestion();
  const { feedback, loading: submitLoading, error: submitError, submit, clearFeedback } = useSubmit();

  // Theme management
  useEffect(() => {
    const savedTheme = localStorage.getItem(config.storageKeys.theme) || 'dark';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  // Check if already started
  useEffect(() => {
    if (sessionStorage.getItem(config.storageKeys.started) === 'true') {
      setStarted(true);
      loadQuestion();
      setQuestionNumber(1);
    }
  }, [loadQuestion]);

  // Show toast for errors
  useEffect(() => {
    if (questionError) {
      showToast(questionError, 'error');
    }
  }, [questionError]);

  useEffect(() => {
    if (submitError) {
      showToast(submitError, 'error');
    }
  }, [submitError]);

  const toggleTheme = useCallback(() => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem(config.storageKeys.theme, newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  }, [theme]);

  const showToast = useCallback((message, type = 'success') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  }, []);

  const handleStart = useCallback(() => {
    sessionStorage.setItem(config.storageKeys.started, 'true');
    setStarted(true);
    setQuestionNumber(1);
    loadQuestion();
  }, [loadQuestion]);

  const handleSubmit = useCallback(async (selectedAnswer, explanation) => {
    if (!question) return;

    if (!selectedAnswer) {
      showToast('Choose one answer option first.', 'error');
      return;
    }

    if (!explanation) {
      showToast('Explain how you solved the question.', 'error');
      return;
    }

    if (explanation.length < config.minExplanationLength) {
      showToast('Please write a little more so the app can judge your reasoning.', 'error');
      return;
    }

    const result = await submit(question.id, selectedAnswer, explanation);
    if (result) {
      addResult(result.is_correct);
    }
  }, [question, submit, addResult, showToast]);

  const handleNext = useCallback(() => {
    clearFeedback();
    setQuestionNumber((n) => n + 1);
    loadQuestion();
  }, [clearFeedback, loadQuestion]);

  const handleSkip = useCallback(() => {
    clearFeedback();
    setQuestionNumber((n) => n + 1);
    loadQuestion();
  }, [clearFeedback, loadQuestion]);

  return (
    <div className="app">
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>
      <div className="floating-orbs">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <Navbar score={score} theme={theme} onToggleTheme={toggleTheme} />

      <main className="main-content">
        {!started ? (
          <Hero onStart={handleStart} />
        ) : (
          <section className="practice-section">
            <div className="practice-container">
              <div className="glass-card practice-guide">
                <div className="guide-steps">
                  <div className="guide-step">
                    <span className="guide-step-number">1</span>
                    <span>Choose one option</span>
                  </div>
                  <div className="guide-step">
                    <span className="guide-step-number">2</span>
                    <span>Explain your reasoning</span>
                  </div>
                  <div className="guide-step">
                    <span className="guide-step-number">3</span>
                    <span>Review the verdict and fix the misconception</span>
                  </div>
                </div>
              </div>

              <QuestionCard
                question={question}
                questionNumber={questionNumber}
                loading={questionLoading}
                onSkip={handleSkip}
              />

              {!feedback && (
                <AnswerForm
                  question={question}
                  loading={submitLoading}
                  onSubmit={handleSubmit}
                />
              )}

              {feedback && (
                <FeedbackPanel
                  feedback={feedback}
                  questionId={question?.id}
                  onNext={handleNext}
                />
              )}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <i className="fas fa-graduation-cap"></i>
            <span>MAP Student Practice</span>
          </div>
          <div className="footer-info">
            <span>MCQ + explanation + misconception feedback</span>
          </div>
        </div>
      </footer>

      <div className="toast-container">
        {toasts.map((toast) => (
          <Toast key={toast.id} message={toast.message} type={toast.type} />
        ))}
      </div>
    </div>
  );
}

export default App;
