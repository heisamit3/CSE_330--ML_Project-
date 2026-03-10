/**
 * MAP — Math Practice Portal
 * Student-facing frontend logic
 */

// ============================================================
// THEME MANAGEMENT
// ============================================================
const ThemeManager = {
    init() {
        const saved = localStorage.getItem('map-student-theme') || 'dark';
        this.set(saved);
        document.getElementById('themeToggle').addEventListener('click', () => this.toggle());
    },

    set(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('map-student-theme', theme);
        const icon = document.querySelector('#themeToggle i');
        icon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    },

    toggle() {
        const current = document.documentElement.getAttribute('data-theme');
        this.set(current === 'dark' ? 'light' : 'dark');
    }
};

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================
const Toast = {
    show(message, type = 'success') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <i class="toast-icon fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
            <span>${message}</span>
        `;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = 'toastOut 0.3s ease-in forwards';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
};

// ============================================================
// SCORE TRACKER
// ============================================================
const Score = {
    correct: 0,
    wrong: 0,
    total: 0,

    init() {
        // Load from session
        const saved = sessionStorage.getItem('map-student-score');
        if (saved) {
            const data = JSON.parse(saved);
            this.correct = data.correct || 0;
            this.wrong = data.wrong || 0;
            this.total = data.total || 0;
        }
        this.render();
    },

    add(isCorrect) {
        this.total++;
        if (isCorrect) this.correct++;
        else this.wrong++;
        this.save();
        this.render();
    },

    save() {
        sessionStorage.setItem('map-student-score', JSON.stringify({
            correct: this.correct,
            wrong: this.wrong,
            total: this.total,
        }));
    },

    render() {
        document.getElementById('correctCount').textContent = this.correct;
        document.getElementById('wrongCount').textContent = this.wrong;
        document.getElementById('totalCount').textContent = this.total;
    }
};

// ============================================================
// PRACTICE APP
// ============================================================
const Practice = {
    currentQuestion: null,
    questionNumber: 0,
    isSubmitting: false,

    init() {
        // Start button
        document.getElementById('startBtn').addEventListener('click', () => this.start());

        // Submit form
        document.getElementById('answerForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submit();
        });

        // Skip button
        document.getElementById('skipBtn').addEventListener('click', () => this.loadQuestion());

        // Next button
        document.getElementById('nextBtn').addEventListener('click', () => this.nextQuestion());

        // Character counter
        const textarea = document.getElementById('studentExplanation');
        textarea.addEventListener('input', () => {
            document.getElementById('charCount').textContent = textarea.value.length;
        });

        // Star rating
        document.querySelectorAll('.star-btn').forEach(btn => {
            btn.addEventListener('click', () => this.rate(parseInt(btn.dataset.rating)));
        });

        // Check if already started (page refresh)
        const inProgress = sessionStorage.getItem('map-student-started');
        if (inProgress === 'true') {
            this.start();
        }
    },

    async start() {
        sessionStorage.setItem('map-student-started', 'true');
        document.getElementById('welcomeHero').style.display = 'none';
        document.getElementById('practiceSection').style.display = 'block';
        await this.loadQuestion();
    },

    async loadQuestion() {
        // Reset UI
        document.getElementById('feedbackPanel').style.display = 'none';
        document.getElementById('answerCard').style.display = 'block';
        document.getElementById('studentAnswer').value = '';
        document.getElementById('studentExplanation').value = '';
        document.getElementById('charCount').textContent = '0';
        document.getElementById('submitBtn').disabled = false;

        // Reset star rating
        document.querySelectorAll('.star-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('ratingThanks').style.display = 'none';
        document.getElementById('starRating').style.display = 'flex';

        this.questionNumber++;
        document.getElementById('questionNumber').textContent = this.questionNumber;

        try {
            const res = await fetch('/api/question');
            const data = await res.json();

            if (data.error) {
                document.getElementById('questionText').textContent = 'Error loading question. Please try again.';
                return;
            }

            this.currentQuestion = data;
            document.getElementById('questionText').innerHTML = this.formatQuestionText(data.text);

            // Render LaTeX if available
            if (window.renderMathInElement) {
                renderMathInElement(document.getElementById('questionText'), {
                    delimiters: [
                        { left: '\\(', right: '\\)', display: false },
                        { left: '\\[', right: '\\]', display: true },
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                    ],
                    throwOnError: false,
                });
            }

            // Scroll to question
            document.getElementById('questionCard').scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (e) {
            console.error('Failed to load question:', e);
            document.getElementById('questionText').textContent = 'Failed to load question. Please check your connection.';
        }
    },

    formatQuestionText(text) {
        // Highlight [Image: ...] descriptions
        return text.replace(
            /\[Image: (.*?)\]/gi,
            '<div class="image-note"><i class="fas fa-image"></i> <em>$1</em></div>'
        );
    },

    async submit() {
        if (this.isSubmitting || !this.currentQuestion) return;

        const answer = document.getElementById('studentAnswer').value.trim();
        const explanation = document.getElementById('studentExplanation').value.trim();

        if (!answer) {
            Toast.show('Please enter your answer.', 'error');
            return;
        }
        if (!explanation) {
            Toast.show('Please explain your reasoning.', 'error');
            return;
        }
        if (explanation.length < 10) {
            Toast.show('Please provide a more detailed explanation (at least 10 characters).', 'error');
            return;
        }

        this.isSubmitting = true;
        this.showLoading();

        try {
            const res = await fetch('/api/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question_id: this.currentQuestion.id,
                    student_answer: answer,
                    student_explanation: explanation,
                }),
            });

            if (!res.ok) throw new Error(`Server error: ${res.status}`);

            const data = await res.json();

            if (data.success) {
                Score.add(data.feedback.is_correct);
                this.showFeedback(data.feedback);
            } else {
                throw new Error(data.error || 'Submission failed');
            }
        } catch (e) {
            Toast.show(`Error: ${e.message}`, 'error');
            this.resetSubmitButton();
        } finally {
            this.isSubmitting = false;
        }
    },

    showLoading() {
        const btn = document.getElementById('submitBtn');
        btn.disabled = true;
        btn.querySelector('span').style.display = 'none';
        btn.querySelector('.btn-loader').style.display = 'flex';
    },

    resetSubmitButton() {
        const btn = document.getElementById('submitBtn');
        btn.disabled = false;
        btn.querySelector('span').style.display = 'inline';
        btn.querySelector('.btn-loader').style.display = 'none';
    },

    showFeedback(feedback) {
        this.resetSubmitButton();

        // Hide answer form, show feedback
        document.getElementById('answerCard').style.display = 'none';
        document.getElementById('feedbackPanel').style.display = 'block';

        // Result Banner
        const iconWrap = document.getElementById('resultIconWrap');
        const icon = document.getElementById('resultIcon');
        const headline = document.getElementById('resultHeadline');
        const message = document.getElementById('resultMessage');

        iconWrap.className = 'result-icon-wrap';
        if (feedback.is_correct) {
            iconWrap.classList.add('correct');
            icon.className = 'fas fa-check-circle';
        } else {
            iconWrap.classList.add('incorrect');
            icon.className = 'fas fa-times-circle';
        }

        // Apply color to headline
        headline.textContent = feedback.result_headline;
        headline.style.color = feedback.result_color;
        message.textContent = feedback.result_message;

        // Answer comparison
        document.getElementById('yourAnswerDisplay').textContent = feedback.student_answer;
        const correctDisplay = document.getElementById('correctAnswerDisplay');
        correctDisplay.textContent = feedback.correct_answer;

        // Render LaTeX in answers
        if (window.renderMathInElement) {
            renderMathInElement(correctDisplay, {
                delimiters: [
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                ],
                throwOnError: false,
            });
        }

        // Correct solution
        const solutionEl = document.getElementById('solutionContent');
        if (feedback.correct_explanation) {
            solutionEl.textContent = feedback.correct_explanation;
        } else {
            solutionEl.textContent = 'The correct answer is ' + feedback.correct_answer + '.';
        }

        // Misconception card
        const misconCard = document.getElementById('misconceptionCard');
        if (feedback.has_misconception) {
            misconCard.style.display = 'block';
            document.getElementById('misconceptionName').textContent = feedback.misconception_name;

            const whatWrong = document.getElementById('whatWentWrong');
            if (feedback.what_went_wrong) {
                whatWrong.textContent = feedback.what_went_wrong;
                whatWrong.parentElement.style.display = 'block';
            } else {
                whatWrong.parentElement.style.display = 'none';
            }

            document.getElementById('misconceptionTip').textContent = feedback.misconception_tip;
        } else {
            misconCard.style.display = 'none';
        }

        // Scroll to feedback
        setTimeout(() => {
            document.getElementById('feedbackPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    },

    nextQuestion() {
        this.loadQuestion();
        // Scroll to top of practice section
        document.getElementById('practiceSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    async rate(rating) {
        // Highlight stars
        document.querySelectorAll('.star-btn').forEach(btn => {
            const r = parseInt(btn.dataset.rating);
            if (r <= rating) btn.classList.add('active');
            else btn.classList.remove('active');
        });

        try {
            await fetch('/api/rate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question_id: this.currentQuestion?.id || 'unknown',
                    rating: rating,
                    comment: null,
                }),
            });

            setTimeout(() => {
                document.getElementById('starRating').style.display = 'none';
                document.getElementById('ratingThanks').style.display = 'flex';
            }, 500);

            Toast.show('Thanks for your feedback!', 'success');
        } catch (e) {
            console.warn('Could not send rating:', e);
        }
    }
};

// ============================================================
// INITIALIZE
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    Score.init();
    Practice.init();

    // Initialize KaTeX auto-render if loaded
    if (window.renderMathInElement) {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true },
            ],
            throwOnError: false,
        });
    }
});
