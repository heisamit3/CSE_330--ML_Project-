import React, { useState, useRef, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { useNotifications } from '../hooks/useNotifications';
import { supabase } from '../services/supabase';
import { formatQuestionText } from '../utils/math';
import './NotificationBell.css';

function NotificationBell() {
  const { notifications, unreadCount, loading, markAsRead, markAllAsRead } = useNotifications();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedNotification, setSelectedNotification] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackError, setFeedbackError] = useState('');
  const menuRef = useRef(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    function handleEscape(event) {
      if (event.key === 'Escape') {
        setSelectedNotification(null);
        setFeedbackError('');
      }
    }

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const handleNotificationClick = async (notification) => {
    if (!notification.is_read) {
      await markAsRead(notification.id);
    }

    setIsOpen(false);
    setFeedbackLoading(true);
    setFeedbackError('');

    try {
      if (!notification.submission_id || !notification.feedback_id) {
        throw new Error('This notification is missing its feedback reference.');
      }

      const [{ data: submission, error: submissionError }, { data: feedback, error: feedbackFetchError }] = await Promise.all([
        supabase
          .from('submissions')
          .select('id, question_text, selected_answer, student_explanation, is_correct, created_at')
          .eq('id', notification.submission_id)
          .single(),
        supabase
          .from('teacher_feedback')
          .select('id, feedback_text, additional_prediction, correct_prediction_label, created_at')
          .eq('id', notification.feedback_id)
          .single(),
      ]);

      if (submissionError) throw submissionError;
      if (feedbackFetchError) throw feedbackFetchError;

      setSelectedNotification({
        notification,
        submission,
        feedback,
      });
    } catch (err) {
      console.error('Error loading notification feedback:', err);
      setFeedbackError(err.message || 'Failed to load feedback details.');
      setSelectedNotification({
        notification,
        submission: null,
        feedback: null,
      });
    } finally {
      setFeedbackLoading(false);
    }
  };

  return (
    <>
      <div className="notification-bell" ref={menuRef}>
        <button
          className="bell-button"
          onClick={() => setIsOpen(!isOpen)}
          title="Notifications"
        >
          <i className="fas fa-bell"></i>
          {unreadCount > 0 && (
            <span className="notification-badge">{unreadCount > 9 ? '9+' : unreadCount}</span>
          )}
        </button>

        {isOpen && (
          <div className="notification-menu">
            <div className="notification-header">
              <h3>Notifications</h3>
              {unreadCount > 0 && (
                <button className="mark-all-read" onClick={markAllAsRead}>
                  Mark all read
                </button>
              )}
            </div>

            <div className="notification-list">
              {loading ? (
                <div className="notification-loading">Loading...</div>
              ) : notifications.length === 0 ? (
                <div className="notification-empty">
                  <i className="fas fa-bell-slash"></i>
                  <p>No notifications</p>
                </div>
              ) : (
                notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`notification-item ${notification.is_read ? 'read' : 'unread'}`}
                    onClick={() => handleNotificationClick(notification)}
                  >
                    <div className="notification-icon">
                      <i className="fas fa-comment-dots"></i>
                    </div>
                    <div className="notification-content">
                      <p className="notification-title">{notification.title}</p>
                      <p className="notification-message">{notification.message}</p>
                      <span className="notification-time">{formatDate(notification.created_at)}</span>
                    </div>
                    {!notification.is_read && <span className="unread-dot" />}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      {selectedNotification && ReactDOM.createPortal(
        <div className="notification-modal-backdrop" onClick={() => setSelectedNotification(null)}>
          <div
            className="notification-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="notification-modal-header">
              <div>
                <h3>{selectedNotification.notification.title}</h3>
                <p>{selectedNotification.notification.message}</p>
              </div>
              <button
                className="notification-modal-close"
                onClick={() => setSelectedNotification(null)}
                title="Close"
              >
                <i className="fas fa-times"></i>
              </button>
            </div>

            <div className="notification-modal-body">
              {feedbackLoading ? (
                <div className="notification-feedback-state">Loading feedback...</div>
              ) : feedbackError ? (
                <div className="notification-feedback-state error">{feedbackError}</div>
              ) : (
                <>
                  <div className="notification-feedback-section">
                    <span className="notification-feedback-label">Question</span>
                    <div
                      className="notification-feedback-value"
                      dangerouslySetInnerHTML={{
                        __html: formatQuestionText(selectedNotification.submission?.question_text || ''),
                      }}
                    />
                  </div>

                  <div className="notification-feedback-grid">
                    <div className="notification-feedback-section">
                      <span className="notification-feedback-label">Your answer</span>
                      <p className="notification-feedback-value">
                        {selectedNotification.submission?.selected_answer || 'Not available'}
                      </p>
                    </div>

                    <div className="notification-feedback-section">
                      <span className="notification-feedback-label">Result</span>
                      <p className="notification-feedback-value">
                        {selectedNotification.submission?.is_correct ? 'Correct' : 'Incorrect'}
                      </p>
                    </div>
                  </div>

                  <div className="notification-feedback-section">
                    <span className="notification-feedback-label">Your explanation</span>
                    <p className="notification-feedback-value">
                      {selectedNotification.submission?.student_explanation || 'Not available'}
                    </p>
                  </div>

                  <div className="notification-feedback-section highlight">
                    <span className="notification-feedback-label">Teacher feedback</span>
                    <p className="notification-feedback-value">
                      {selectedNotification.feedback?.feedback_text || 'No feedback text available.'}
                    </p>
                  </div>

                  {selectedNotification.feedback?.additional_prediction && (
                    <div className="notification-feedback-section">
                      <span className="notification-feedback-label">Additional teacher note</span>
                      <p className="notification-feedback-value">
                        {selectedNotification.feedback.additional_prediction}
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

export default NotificationBell;
