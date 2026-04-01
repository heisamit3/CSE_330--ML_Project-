import React, { useState, useEffect } from 'react';

// These are the actual misconception labels from the trained model (label_encoder.pkl)
// Using only the misconception name without True/False prefix
const MISCONCEPTION_LABELS = [
  'Adding_across',
  'Adding_terms',
  'Additive',
  'Base_rate',
  'Certainty',
  'Definition',
  'Denominator-only_change',
  'Division',
  'Duplication',
  'Firstterm',
  'FlipChange',
  'Ignores_zeroes',
  'Incomplete',
  'Incorrect_equivalent_fraction_addition',
  'Interior',
  'Inverse_operation',
  'Inversion',
  'Irrelevant',
  'Longer_is_bigger',
  'Mult',
  'Multiplying_by_4',
  'Not_variable',
  'Positive',
  'Scale',
  'Shorter_is_bigger',
  'Subtraction',
  'SwapDividend',
  'Tacking',
  'Unknowable',
  'WNB',
  'Whole_numbers_larger',
  'Wrong_Fraction',
  'Wrong_Operation',
  'Wrong_fraction',
  'Wrong_term',
];

const CATEGORIES = [
  'Fractions',
  'Decimals',
  'Percentages',
  'Algebra',
  'Geometry',
  'Ratios',
  'Probability',
  'Statistics',
  'Arithmetic',
  'Word Problems',
  'Other',
];

const DIFFICULTIES = ['easy', 'medium', 'hard'];

function QuestionForm({ question, onSubmit, onCancel, loading }) {
  const [formData, setFormData] = useState({
    question_text: '',
    option_a: '',
    option_b: '',
    option_c: '',
    option_d: '',
    correct_answer: 'A',
    expected_misconceptions: [],
    category: '',
    difficulty: 'medium',
    status: 'draft',
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (question) {
      setFormData({
        question_text: question.question_text || '',
        option_a: question.option_a || '',
        option_b: question.option_b || '',
        option_c: question.option_c || '',
        option_d: question.option_d || '',
        correct_answer: question.correct_answer || 'A',
        expected_misconceptions: question.expected_misconceptions || [],
        category: question.category || '',
        difficulty: question.difficulty || 'medium',
        status: question.status || 'draft',
      });
    }
  }, [question]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: null }));
    }
  };

  const handleMisconceptionToggle = (misconception) => {
    setFormData((prev) => {
      const current = prev.expected_misconceptions;
      if (current.includes(misconception)) {
        return {
          ...prev,
          expected_misconceptions: current.filter((m) => m !== misconception),
        };
      }
      return {
        ...prev,
        expected_misconceptions: [...current, misconception],
      };
    });
  };

  const validate = () => {
    const newErrors = {};

    if (!formData.question_text.trim()) {
      newErrors.question_text = 'Question text is required';
    }
    if (!formData.option_a.trim()) {
      newErrors.option_a = 'Option A is required';
    }
    if (!formData.option_b.trim()) {
      newErrors.option_b = 'Option B is required';
    }
    if (formData.expected_misconceptions.length === 0) {
      newErrors.expected_misconceptions = 'Select at least one expected misconception';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      onSubmit(formData);
    }
  };

  return (
    <form className="question-form" onSubmit={handleSubmit}>
      <div className="form-section">
        <h3>
          <i className="fas fa-question-circle"></i>
          Question Details
        </h3>

        <div className="form-group">
          <label htmlFor="question_text">
            Question Text <span className="required">*</span>
          </label>
          <textarea
            id="question_text"
            name="question_text"
            value={formData.question_text}
            onChange={handleChange}
            placeholder="Enter the question text. You can use LaTeX notation like \( \frac{1}{2} \) for math expressions."
            rows={4}
            disabled={loading}
            className={errors.question_text ? 'error' : ''}
          />
          {errors.question_text && (
            <span className="error-message">{errors.question_text}</span>
          )}
          <span className="hint">Supports LaTeX: \( ... \) for inline math</span>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="category">Category</label>
            <select
              id="category"
              name="category"
              value={formData.category}
              onChange={handleChange}
              disabled={loading}
            >
              <option value="">-- Select Category --</option>
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="difficulty">Difficulty</label>
            <select
              id="difficulty"
              name="difficulty"
              value={formData.difficulty}
              onChange={handleChange}
              disabled={loading}
            >
              {DIFFICULTIES.map((diff) => (
                <option key={diff} value={diff}>
                  {diff.charAt(0).toUpperCase() + diff.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>
          <i className="fas fa-list-ol"></i>
          Answer Options
        </h3>

        <div className="options-grid">
          {['A', 'B', 'C', 'D'].map((letter) => (
            <div key={letter} className="option-input-group">
              <div className="option-header">
                <label htmlFor={`option_${letter.toLowerCase()}`}>
                  Option {letter}
                  {letter === 'A' || letter === 'B' ? (
                    <span className="required">*</span>
                  ) : (
                    <span className="optional">(optional)</span>
                  )}
                </label>
                <label className="correct-radio">
                  <input
                    type="radio"
                    name="correct_answer"
                    value={letter}
                    checked={formData.correct_answer === letter}
                    onChange={handleChange}
                    disabled={loading}
                  />
                  <span className="radio-label">Correct</span>
                </label>
              </div>
              <input
                type="text"
                id={`option_${letter.toLowerCase()}`}
                name={`option_${letter.toLowerCase()}`}
                value={formData[`option_${letter.toLowerCase()}`]}
                onChange={handleChange}
                placeholder={`Enter option ${letter} (LaTeX supported)`}
                disabled={loading}
                className={errors[`option_${letter.toLowerCase()}`] ? 'error' : ''}
              />
              {errors[`option_${letter.toLowerCase()}`] && (
                <span className="error-message">
                  {errors[`option_${letter.toLowerCase()}`]}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="form-section">
        <h3>
          <i className="fas fa-brain"></i>
          Expected Misconceptions <span className="required">*</span>
        </h3>
        <p className="section-hint">
          Select the misconceptions students might exhibit when answering this question
          incorrectly. This helps the AI model during classification.
        </p>

        {errors.expected_misconceptions && (
          <div className="error-message section-error">
            {errors.expected_misconceptions}
          </div>
        )}

        <div className="misconceptions-grid">
          {MISCONCEPTION_LABELS.map((misconception) => (
            <label
              key={misconception}
              className={`misconception-chip ${
                formData.expected_misconceptions.includes(misconception)
                  ? 'selected'
                  : ''
              }`}
            >
              <input
                type="checkbox"
                checked={formData.expected_misconceptions.includes(misconception)}
                onChange={() => handleMisconceptionToggle(misconception)}
                disabled={loading}
              />
              <span>{misconception.replace(/_/g, ' ')}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="form-section">
        <h3>
          <i className="fas fa-cog"></i>
          Status
        </h3>

        <div className="status-options">
          {[
            { value: 'draft', label: 'Draft', desc: 'Not visible to students' },
            { value: 'active', label: 'Active', desc: 'Visible to students' },
            { value: 'archived', label: 'Archived', desc: 'Hidden from all views' },
          ].map((opt) => (
            <label
              key={opt.value}
              className={`status-option ${
                formData.status === opt.value ? 'selected' : ''
              }`}
            >
              <input
                type="radio"
                name="status"
                value={opt.value}
                checked={formData.status === opt.value}
                onChange={handleChange}
                disabled={loading}
              />
              <div className="status-content">
                <span className="status-label">{opt.label}</span>
                <span className="status-desc">{opt.desc}</span>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div className="form-actions">
        <button
          type="button"
          className="btn-secondary"
          onClick={onCancel}
          disabled={loading}
        >
          Cancel
        </button>
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? (
            <>
              <i className="fas fa-spinner fa-spin"></i>
              Saving...
            </>
          ) : (
            <>
              <i className="fas fa-save"></i>
              {question ? 'Update Question' : 'Create Question'}
            </>
          )}
        </button>
      </div>
    </form>
  );
}

export default QuestionForm;
