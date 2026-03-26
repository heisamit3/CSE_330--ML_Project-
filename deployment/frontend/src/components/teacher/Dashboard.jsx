import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../context/AuthContext';
import { supabase } from '../../services/supabase';
import SubmissionList from './SubmissionList';
import SubmissionDetail from './SubmissionDetail';
import QuestionManager from './QuestionManager';

function Dashboard() {
  const { profile } = useAuth();
  const [activeTab, setActiveTab] = useState('submissions'); // 'submissions' or 'questions'
  const [submissions, setSubmissions] = useState([]);
  const [selectedSubmission, setSelectedSubmission] = useState(null);
  const [filter, setFilter] = useState('pending'); // 'pending', 'reviewed', 'all'
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, pending: 0, reviewed: 0 });

  const loadSubmissions = useCallback(async () => {
    setLoading(true);
    try {
      let query = supabase
        .from('submissions')
        .select(`
          *,
          student:profiles!submissions_student_id_fkey(display_name, email),
          teacher_feedback(*)
        `)
        .order('created_at', { ascending: false });

      const { data, error } = await query;

      if (error) throw error;

      // Calculate stats
      const total = data?.length || 0;
      const reviewed = data?.filter(s => s.teacher_feedback?.length > 0).length || 0;
      const pending = total - reviewed;
      setStats({ total, pending, reviewed });

      // Filter based on current filter
      let filtered = data || [];
      if (filter === 'pending') {
        filtered = filtered.filter(s => !s.teacher_feedback?.length);
      } else if (filter === 'reviewed') {
        filtered = filtered.filter(s => s.teacher_feedback?.length > 0);
      }

      setSubmissions(filtered);
    } catch (err) {
      console.error('Error loading submissions:', err);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    loadSubmissions();
  }, [loadSubmissions]);

  const handleFeedbackSubmitted = () => {
    loadSubmissions();
    setSelectedSubmission(null);
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="dashboard-title">
          <h1>Teacher Dashboard</h1>
          <p>Welcome, {profile?.display_name || 'Teacher'}</p>
        </div>
        {activeTab === 'submissions' && (
          <div className="dashboard-stats">
            <div className="stat-card">
              <span className="stat-value">{stats.total}</span>
              <span className="stat-label">Total</span>
            </div>
            <div className="stat-card pending">
              <span className="stat-value">{stats.pending}</span>
              <span className="stat-label">Pending</span>
            </div>
            <div className="stat-card reviewed">
              <span className="stat-value">{stats.reviewed}</span>
              <span className="stat-label">Reviewed</span>
            </div>
          </div>
        )}
      </header>

      <div className="dashboard-tabs">
        <button
          className={`tab-btn ${activeTab === 'submissions' ? 'active' : ''}`}
          onClick={() => setActiveTab('submissions')}
        >
          <i className="fas fa-inbox"></i>
          Student Submissions
        </button>
        <button
          className={`tab-btn ${activeTab === 'questions' ? 'active' : ''}`}
          onClick={() => setActiveTab('questions')}
        >
          <i className="fas fa-question-circle"></i>
          Question Bank
        </button>
      </div>

      {activeTab === 'submissions' ? (
        <>
          <div className="dashboard-filters">
            <button
              className={`filter-btn ${filter === 'pending' ? 'active' : ''}`}
              onClick={() => setFilter('pending')}
            >
              <i className="fas fa-clock"></i>
              Pending ({stats.pending})
            </button>
            <button
              className={`filter-btn ${filter === 'reviewed' ? 'active' : ''}`}
              onClick={() => setFilter('reviewed')}
            >
              <i className="fas fa-check"></i>
              Reviewed ({stats.reviewed})
            </button>
            <button
              className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              <i className="fas fa-list"></i>
              All ({stats.total})
            </button>
          </div>

          <div className="dashboard-content">
            <div className={`dashboard-list ${selectedSubmission ? 'with-detail' : ''}`}>
              <SubmissionList
                submissions={submissions}
                loading={loading}
                onSelect={setSelectedSubmission}
                selected={selectedSubmission}
              />
            </div>

            {selectedSubmission && (
              <div className="dashboard-detail">
                <SubmissionDetail
                  submission={selectedSubmission}
                  onFeedbackSubmitted={handleFeedbackSubmitted}
                  onClose={() => setSelectedSubmission(null)}
                />
              </div>
            )}
          </div>
        </>
      ) : (
        <QuestionManager />
      )}
    </div>
  );
}

export default Dashboard;
