import React, { useState, useEffect, useCallback } from 'react';
import QuestionCard from '../components/QuestionCard';
import AnswerForm from '../components/AnswerForm';
import FeedbackPanel from '../components/FeedbackPanel';
import Hero from '../components/Hero';
import Toast from '../components/Toast';
import { useScore } from '../hooks/useScore';
import { useQuestion } from '../hooks/useQuestion';
import { useSubmit } from '../hooks/useSubmit';
import config from '../config';

function StudentPracticePage() {
  const [started, setStarted] = useState(false);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [toasts, setToasts] = useState([]);

  const { addResult } = useScore();
  const {
    question,
    loading: questionLoading,
    error: questionError,
    loadQuestion,
    noMoreQuestions,
  } = useQuestion();
  const { feedback, loading: submitLoading, error: submitError, submit, clearFeedback } = useSubmit();

  // Check if already started
  useEffect(() => {
    if (sessionStorage.getItem(config.storageKeys.started) === 'true') {
      setStarted(true);
      loadQuestion();
      setQuestionNumber(1);
    }
  }, [loadQuestion]);

  const showToast = useCallback((message, type = 'success') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  }, []);

  // Show toast for errors
  useEffect(() => {
    if (questionError) {
      showToast(questionError, 'error');
    }
  }, [questionError, showToast]);

  useEffect(() => {
    if (submitError) {
      showToast(submitError, 'error');
    }
  }, [submitError, showToast]);

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

    const result = await submit(question, selectedAnswer, explanation);
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
    <>
      {!started ? (
        <Hero onStart={handleStart} />
      ) : noMoreQuestions ? (
        <section className="practice-section">
          <div className="practice-container">
            <div className="glass-card completed-card">
              <div className="completed-icon">
                <i className="fas fa-trophy"></i>
              </div>
              <h2>All Questions Completed!</h2>
              <p>
                Great job! You have answered all available questions.
                Check back later for new questions from your teachers.
              </p>
              <div className="completed-actions">
                <button
                  className="btn-secondary"
                  onClick={() => {
                    sessionStorage.removeItem(config.storageKeys.started);
                    setStarted(false);
                    setQuestionNumber(0);
                  }}
                >
                  <i className="fas fa-home"></i>
                  Back to Home
                </button>
              </div>
            </div>
          </div>
        </section>
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

      <div className="toast-container">
        {toasts.map((toast) => (
          <Toast key={toast.id} message={toast.message} type={toast.type} />
        ))}
      </div>
    </>
  );
}

export default StudentPracticePage;
