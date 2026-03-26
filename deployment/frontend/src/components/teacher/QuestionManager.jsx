import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../context/AuthContext';
import { supabase } from '../../services/supabase';
import QuestionForm from './QuestionForm';

function QuestionManager() {
  const { profile } = useAuth();
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // 'all', 'draft', 'active', 'archived', 'mine'
  const [showForm, setShowForm] = useState(false);
  const [editingQuestion, setEditingQuestion] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ total: 0, draft: 0, active: 0, archived: 0 });

  const loadQuestions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let query = supabase
        .from('teacher_questions')
        .select(`
          *,
          teacher:profiles!teacher_questions_teacher_id_fkey(display_name, email)
        `)
        .order('created_at', { ascending: false });

      const { data, error: fetchError } = await query;

      if (fetchError) throw fetchError;

      // Calculate stats
      const total = data?.length || 0;
      const draft = data?.filter((q) => q.status === 'draft').length || 0;
      const active = data?.filter((q) => q.status === 'active').length || 0;
      const archived = data?.filter((q) => q.status === 'archived').length || 0;
      setStats({ total, draft, active, archived });

      // Apply filter
      let filtered = data || [];
      if (filter === 'draft') {
        filtered = filtered.filter((q) => q.status === 'draft');
      } else if (filter === 'active') {
        filtered = filtered.filter((q) => q.status === 'active');
      } else if (filter === 'archived') {
        filtered = filtered.filter((q) => q.status === 'archived');
      } else if (filter === 'mine') {
        filtered = filtered.filter((q) => q.teacher_id === profile?.id);
      }

      setQuestions(filtered);
    } catch (err) {
      console.error('Error loading questions:', err);
      setError('Failed to load questions');
    } finally {
      setLoading(false);
    }
  }, [filter, profile?.id]);

  useEffect(() => {
    loadQuestions();
  }, [loadQuestions]);

  const handleCreateQuestion = async (formData) => {
    setSaving(true);
    setError(null);
    try {
      const { error: insertError } = await supabase.from('teacher_questions').insert({
        teacher_id: profile.id,
        question_text: formData.question_text,
        option_a: formData.option_a,
        option_b: formData.option_b,
        option_c: formData.option_c || null,
        option_d: formData.option_d || null,
        correct_answer: formData.correct_answer,
        expected_misconceptions: formData.expected_misconceptions,
        category: formData.category || null,
        difficulty: formData.difficulty,
        status: formData.status,
      });

      if (insertError) throw insertError;

      setShowForm(false);
      loadQuestions();
    } catch (err) {
      console.error('Error creating question:', err);
      setError('Failed to create question: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleUpdateQuestion = async (formData) => {
    if (!editingQuestion) return;

    setSaving(true);
    setError(null);
    try {
      const { error: updateError } = await supabase
        .from('teacher_questions')
        .update({
          question_text: formData.question_text,
          option_a: formData.option_a,
          option_b: formData.option_b,
          option_c: formData.option_c || null,
          option_d: formData.option_d || null,
          correct_answer: formData.correct_answer,
          expected_misconceptions: formData.expected_misconceptions,
          category: formData.category || null,
          difficulty: formData.difficulty,
          status: formData.status,
          updated_at: new Date().toISOString(),
        })
        .eq('id', editingQuestion.id);

      if (updateError) throw updateError;

      setEditingQuestion(null);
      loadQuestions();
    } catch (err) {
      console.error('Error updating question:', err);
      setError('Failed to update question: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteQuestion = async (questionId) => {
    if (!window.confirm('Are you sure you want to delete this question?')) {
      return;
    }

    try {
      const { error: deleteError } = await supabase
        .from('teacher_questions')
        .delete()
        .eq('id', questionId);

      if (deleteError) throw deleteError;
      loadQuestions();
    } catch (err) {
      console.error('Error deleting question:', err);
      setError('Failed to delete question: ' + err.message);
    }
  };

  const handleStatusChange = async (questionId, newStatus) => {
    try {
      const { error: updateError } = await supabase
        .from('teacher_questions')
        .update({ status: newStatus, updated_at: new Date().toISOString() })
        .eq('id', questionId);

      if (updateError) throw updateError;
      loadQuestions();
    } catch (err) {
      console.error('Error updating status:', err);
      setError('Failed to update status: ' + err.message);
    }
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  // Show form for creating or editing
  if (showForm || editingQuestion) {
    return (
      <div className="question-manager">
        <div className="manager-header">
          <h2>
            <i className="fas fa-plus-circle"></i>
            {editingQuestion ? 'Edit Question' : 'Create New Question'}
          </h2>
        </div>

        {error && (
          <div className="manager-error">
            <i className="fas fa-exclamation-circle"></i>
            {error}
          </div>
        )}

        <QuestionForm
          question={editingQuestion}
          onSubmit={editingQuestion ? handleUpdateQuestion : handleCreateQuestion}
          onCancel={() => {
            setShowForm(false);
            setEditingQuestion(null);
            setError(null);
          }}
          loading={saving}
        />
      </div>
    );
  }

  return (
    <div className="question-manager">
      <div className="manager-header">
        <div className="manager-title">
          <h2>
            <i className="fas fa-question-circle"></i>
            Question Bank
          </h2>
          <p>Create and manage practice questions for students</p>
        </div>
        <button className="btn-primary" onClick={() => setShowForm(true)}>
          <i className="fas fa-plus"></i>
          Add Question
        </button>
      </div>

      <div className="manager-stats">
        <div className="stat-card">
          <span className="stat-value">{stats.total}</span>
          <span className="stat-label">Total</span>
        </div>
        <div className="stat-card draft">
          <span className="stat-value">{stats.draft}</span>
          <span className="stat-label">Draft</span>
        </div>
        <div className="stat-card active">
          <span className="stat-value">{stats.active}</span>
          <span className="stat-label">Active</span>
        </div>
        <div className="stat-card archived">
          <span className="stat-value">{stats.archived}</span>
          <span className="stat-label">Archived</span>
        </div>
      </div>

      <div className="manager-filters">
        {[
          { key: 'all', label: 'All', icon: 'fas fa-list' },
          { key: 'draft', label: 'Draft', icon: 'fas fa-edit' },
          { key: 'active', label: 'Active', icon: 'fas fa-check-circle' },
          { key: 'archived', label: 'Archived', icon: 'fas fa-archive' },
          { key: 'mine', label: 'My Questions', icon: 'fas fa-user' },
        ].map((f) => (
          <button
            key={f.key}
            className={`filter-btn ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
          >
            <i className={f.icon}></i>
            {f.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="manager-error">
          <i className="fas fa-exclamation-circle"></i>
          {error}
        </div>
      )}

      <div className="questions-list">
        {loading ? (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Loading questions...</p>
          </div>
        ) : questions.length === 0 ? (
          <div className="empty-state">
            <i className="fas fa-inbox"></i>
            <p>No questions found</p>
            <button className="btn-secondary" onClick={() => setShowForm(true)}>
              Create your first question
            </button>
          </div>
        ) : (
          questions.map((question) => (
            <div key={question.id} className="question-card">
              <div className="question-header">
                <div className="question-meta">
                  <span className={`status-badge ${question.status}`}>
                    {question.status}
                  </span>
                  {question.difficulty && (
                    <span className={`difficulty-badge ${question.difficulty}`}>
                      {question.difficulty}
                    </span>
                  )}
                  {question.category && (
                    <span className="category-badge">{question.category}</span>
                  )}
                </div>
                <div className="question-actions">
                  <button
                    className="action-btn edit"
                    onClick={() => setEditingQuestion(question)}
                    title="Edit"
                  >
                    <i className="fas fa-edit"></i>
                  </button>
                  {question.status === 'draft' && (
                    <button
                      className="action-btn activate"
                      onClick={() => handleStatusChange(question.id, 'active')}
                      title="Activate"
                    >
                      <i className="fas fa-check"></i>
                    </button>
                  )}
                  {question.status === 'active' && (
                    <button
                      className="action-btn archive"
                      onClick={() => handleStatusChange(question.id, 'archived')}
                      title="Archive"
                    >
                      <i className="fas fa-archive"></i>
                    </button>
                  )}
                  {question.teacher_id === profile?.id && (
                    <button
                      className="action-btn delete"
                      onClick={() => handleDeleteQuestion(question.id)}
                      title="Delete"
                    >
                      <i className="fas fa-trash"></i>
                    </button>
                  )}
                </div>
              </div>

              <div className="question-text">{question.question_text}</div>

              <div className="question-options">
                {['A', 'B', 'C', 'D'].map((letter) => {
                  const optionKey = `option_${letter.toLowerCase()}`;
                  const optionValue = question[optionKey];
                  if (!optionValue) return null;
                  return (
                    <div
                      key={letter}
                      className={`option ${
                        question.correct_answer === letter ? 'correct' : ''
                      }`}
                    >
                      <span className="option-letter">{letter}</span>
                      <span className="option-text">{optionValue}</span>
                      {question.correct_answer === letter && (
                        <i className="fas fa-check correct-icon"></i>
                      )}
                    </div>
                  );
                })}
              </div>

              {question.expected_misconceptions?.length > 0 && (
                <div className="question-misconceptions">
                  <span className="label">Expected misconceptions:</span>
                  <div className="misconception-tags">
                    {question.expected_misconceptions.map((m) => (
                      <span key={m} className="misconception-tag">
                        {m.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="question-footer">
                <span className="created-by">
                  <i className="fas fa-user"></i>
                  {question.teacher?.display_name || 'Unknown'}
                </span>
                <span className="created-at">
                  <i className="fas fa-calendar"></i>
                  {formatDate(question.created_at)}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default QuestionManager;
