const ThemeManager = {
    init() {
        const savedTheme = localStorage.getItem('map-student-theme') || 'dark';
        this.set(savedTheme);
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


const Score = {
    correct: 0,
    wrong: 0,
    total: 0,

    init() {
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
        this.total += 1;
        if (isCorrect) {
            this.correct += 1;
        } else {
            this.wrong += 1;
        }
        this.save();
        this.render();
    },

    save() {
        sessionStorage.setItem('map-student-score', JSON.stringify({
            correct: this.correct,
            wrong: this.wrong,
            total: this.total
        }));
    },

    render() {
        document.getElementById('correctCount').textContent = this.correct;
        document.getElementById('wrongCount').textContent = this.wrong;
        document.getElementById('totalCount').textContent = this.total;
    }
};


const Practice = {
    currentQuestion: null,
    questionNumber: 0,
    isSubmitting: false,

    init() {
        document.getElementById('startBtn').addEventListener('click', () => this.start());
        document.getElementById('skipBtn').addEventListener('click', () => this.loadQuestion());
        document.getElementById('nextBtn').addEventListener('click', () => this.nextQuestion());
        document.getElementById('answerForm').addEventListener('submit', (event) => {
            event.preventDefault();
            this.submit();
        });

        const textarea = document.getElementById('studentExplanation');
        textarea.addEventListener('input', () => {
            document.getElementById('charCount').textContent = textarea.value.length;
        });

        document.querySelectorAll('.star-btn').forEach((button) => {
            button.addEventListener('click', () => this.rate(parseInt(button.dataset.rating, 10)));
        });

        if (sessionStorage.getItem('map-student-started') === 'true') {
            this.start();
        }
    },

    async start() {
        sessionStorage.setItem('map-student-started', 'true');
        document.getElementById('welcomeHero').style.display = 'none';
        document.getElementById('practiceSection').style.display = 'block';

        if (!this.currentQuestion) {
            await this.loadQuestion();
        }
    },

    async loadQuestion() {
        this.resetQuestionUI();
        this.questionNumber += 1;
        document.getElementById('questionNumber').textContent = this.questionNumber;

        try {
            const response = await fetch('/api/question');
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Could not load the question.');
            }

            this.currentQuestion = data;
            document.getElementById('questionText').innerHTML = this.formatQuestionText(data.text);
            this.renderOptions(data.options);
            this.renderMath(document.getElementById('questionText'));
            document.getElementById('questionCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (error) {
            console.error(error);
            document.getElementById('questionText').textContent = 'We could not load a question right now.';
            Toast.show(error.message || 'Failed to load question.', 'error');
        }
    },

    resetQuestionUI() {
        document.getElementById('feedbackPanel').style.display = 'none';
        document.getElementById('answerCard').style.display = 'block';
        document.getElementById('studentExplanation').value = '';
        document.getElementById('charCount').textContent = '0';
        document.getElementById('answerOptions').innerHTML = '';
        this.resetSubmitButton();

        document.querySelectorAll('.star-btn').forEach((button) => button.classList.remove('active'));
        document.getElementById('ratingThanks').style.display = 'none';
        document.getElementById('starRating').style.display = 'flex';
    },

    formatQuestionText(text) {
        return text
            .replace(/\n/g, '<br>')
            .replace(/\[Image:\s*(.*?)\]/gi, '<div class="image-note"><i class="fas fa-image"></i> <em>$1</em></div>');
    },

    renderOptions(options) {
        const container = document.getElementById('answerOptions');
        container.innerHTML = '';

        options.forEach((option) => {
            const label = document.createElement('label');
            label.className = 'option-card';

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'selectedAnswer';
            input.value = option.text;
            input.className = 'option-input';

            const badge = document.createElement('span');
            badge.className = 'option-badge';
            badge.textContent = option.id;

            const content = document.createElement('div');
            content.className = 'option-content';

            const title = document.createElement('div');
            title.className = 'option-title';
            title.textContent = `Option ${option.id}`;

            const body = document.createElement('div');
            body.className = 'option-text';
            body.textContent = option.text;

            content.appendChild(title);
            content.appendChild(body);
            label.appendChild(input);
            label.appendChild(badge);
            label.appendChild(content);
            container.appendChild(label);

            input.addEventListener('change', () => this.syncOptionSelection());
        });

        this.syncOptionSelection();
        this.renderMath(container);
    },

    syncOptionSelection() {
        document.querySelectorAll('.option-card').forEach((card) => {
            const checked = card.querySelector('.option-input')?.checked;
            card.classList.toggle('selected', Boolean(checked));
        });
    },

    getSelectedAnswer() {
        const checked = document.querySelector('input[name="selectedAnswer"]:checked');
        return checked ? checked.value : '';
    },

    async submit() {
        if (this.isSubmitting || !this.currentQuestion) {
            return;
        }

        const selectedAnswer = this.getSelectedAnswer();
        const explanation = document.getElementById('studentExplanation').value.trim();

        if (!selectedAnswer) {
            Toast.show('Choose one answer option first.', 'error');
            return;
        }

        if (!explanation) {
            Toast.show('Explain how you solved the question.', 'error');
            return;
        }

        if (explanation.length < 20) {
            Toast.show('Please write a little more so the app can judge your reasoning.', 'error');
            return;
        }

        this.isSubmitting = true;
        this.showLoading();

        try {
            const response = await fetch('/api/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question_id: this.currentQuestion.id,
                    selected_answer: selectedAnswer,
                    student_explanation: explanation
                })
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Submission failed.');
            }

            Score.add(data.feedback.is_correct);
            this.showFeedback(data.feedback);
        } catch (error) {
            console.error(error);
            Toast.show(error.message || 'Could not submit your answer.', 'error');
            this.resetSubmitButton();
        } finally {
            this.isSubmitting = false;
        }
    },

    showLoading() {
        const button = document.getElementById('submitBtn');
        button.disabled = true;
        button.querySelector('span').style.display = 'none';
        button.querySelector('.btn-loader').style.display = 'flex';
    },

    resetSubmitButton() {
        const button = document.getElementById('submitBtn');
        button.disabled = false;
        button.querySelector('span').style.display = 'inline';
        button.querySelector('.btn-loader').style.display = 'none';
    },

    showFeedback(feedback) {
        this.resetSubmitButton();
        document.getElementById('answerCard').style.display = 'none';
        document.getElementById('feedbackPanel').style.display = 'block';

        const iconWrap = document.getElementById('resultIconWrap');
        const icon = document.getElementById('resultIcon');
        iconWrap.className = 'result-icon-wrap';
        if (feedback.is_correct) {
            iconWrap.classList.add('correct');
            icon.className = 'fas fa-check-circle';
        } else {
            iconWrap.classList.add('incorrect');
            icon.className = 'fas fa-times-circle';
        }

        const headline = document.getElementById('resultHeadline');
        headline.textContent = feedback.result_headline;
        headline.style.color = feedback.result_color;
        document.getElementById('resultMessage').textContent = feedback.result_message;

        const answerVerdict = document.getElementById('answerVerdict');
        const reasoningVerdict = document.getElementById('reasoningVerdict');
        answerVerdict.textContent = feedback.answer_label;
        reasoningVerdict.textContent = feedback.reasoning_label;
        answerVerdict.className = `meta-pill ${feedback.is_correct ? 'meta-pill-correct' : 'meta-pill-incorrect'}`;
        reasoningVerdict.className = `meta-pill ${feedback.has_misconception ? 'meta-pill-warning' : 'meta-pill-neutral'}`;

        const yourAnswer = document.getElementById('yourAnswerDisplay');
        const correctAnswer = document.getElementById('correctAnswerDisplay');
        yourAnswer.textContent = feedback.selected_answer;
        correctAnswer.textContent = feedback.correct_answer;
        this.renderMath(yourAnswer);
        this.renderMath(correctAnswer);

        const solution = document.getElementById('solutionContent');
        solution.textContent = feedback.correct_explanation;
        this.renderMath(solution);

        const misconceptionCard = document.getElementById('misconceptionCard');
        const whatWentWrongSection = document.getElementById('whatWentWrongSection');
        if (feedback.has_misconception) {
            misconceptionCard.style.display = 'block';
            document.getElementById('misconceptionName').textContent = feedback.misconception_name;
            document.getElementById('misconceptionTip').textContent = feedback.misconception_tip;

            if (feedback.what_went_wrong) {
                whatWentWrongSection.style.display = 'block';
                document.getElementById('whatWentWrong').textContent = feedback.what_went_wrong;
            } else {
                whatWentWrongSection.style.display = 'none';
            }
        } else {
            misconceptionCard.style.display = 'none';
        }

        setTimeout(() => {
            document.getElementById('feedbackPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    },

    nextQuestion() {
        this.currentQuestion = null;
        this.loadQuestion();
        document.getElementById('practiceSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    async rate(rating) {
        document.querySelectorAll('.star-btn').forEach((button) => {
            const buttonRating = parseInt(button.dataset.rating, 10);
            button.classList.toggle('active', buttonRating <= rating);
        });

        try {
            await fetch('/api/rate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question_id: this.currentQuestion?.id || 'unknown',
                    rating,
                    comment: null
                })
            });

            setTimeout(() => {
                document.getElementById('starRating').style.display = 'none';
                document.getElementById('ratingThanks').style.display = 'flex';
            }, 400);
        } catch (error) {
            console.warn('Could not send rating:', error);
        }
    },

    renderMath(target) {
        if (!target || !window.renderMathInElement) {
            return;
        }

        renderMathInElement(target, {
            delimiters: [
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true },
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
            ],
            throwOnError: false
        });
    }
};


document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    Score.init();
    Practice.init();
});
