/**
 * MAP — Math Misconception Detector
 * Frontend Application Logic
 */

// ============================================================
// THEME MANAGEMENT
// ============================================================
const ThemeManager = {
    init() {
        const saved = localStorage.getItem('map-theme') || 'dark';
        this.set(saved);
        document.getElementById('themeToggle').addEventListener('click', () => this.toggle());
    },

    set(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('map-theme', theme);
        const icon = document.querySelector('#themeToggle i');
        icon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    },

    toggle() {
        const current = document.documentElement.getAttribute('data-theme');
        this.set(current === 'dark' ? 'light' : 'dark');
    }
};

// ============================================================
// NAVIGATION
// ============================================================
const Navigation = {
    init() {
        document.querySelectorAll('.nav-link[data-section]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.showSection(link.dataset.section);
            });
        });

        // Handle anchor links
        document.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', (e) => {
                const target = link.getAttribute('href').substring(1);
                if (['analyzer', 'dashboard', 'about'].includes(target)) {
                    e.preventDefault();
                    this.showSection(target);
                }
            });
        });
    },

    showSection(sectionId) {
        // Update nav links
        document.querySelectorAll('.nav-link[data-section]').forEach(l => l.classList.remove('active'));
        const activeLink = document.querySelector(`.nav-link[data-section="${sectionId}"]`);
        if (activeLink) activeLink.classList.add('active');

        // Show/hide sections
        ['analyzer', 'dashboard', 'about'].forEach(id => {
            const section = document.getElementById(id);
            if (section) {
                section.style.display = id === sectionId ? 'block' : 'none';
            }
        });

        // Show hero only on analyzer
        const hero = document.getElementById('hero');
        if (hero) {
            hero.style.display = sectionId === 'analyzer' ? 'grid' : 'none';
        }

        // Scroll to top of section
        const target = document.getElementById(sectionId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Load dashboard data if needed
        if (sectionId === 'dashboard') {
            Dashboard.load();
        }
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
// ANALYZER
// ============================================================
const Analyzer = {
    examples: [],
    isLoading: false,

    init() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyze();
        });

        // Load example button
        document.getElementById('loadExampleBtn').addEventListener('click', () => this.loadExample());

        // Answer choice buttons
        document.querySelectorAll('.choice-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.choice-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('answerInput').value = btn.dataset.choice;
            });
        });

        // Character counter
        const explInput = document.getElementById('explanationInput');
        explInput.addEventListener('input', () => {
            document.getElementById('charCount').textContent = explInput.value.length;
        });

        // Feedback buttons
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', () => this.sendFeedback(btn.dataset.feedback === 'helpful'));
        });

        // Load examples
        this.fetchExamples();
    },

    async fetchExamples() {
        try {
            const res = await fetch('/api/examples');
            const data = await res.json();
            this.examples = data.examples || [];
        } catch (e) {
            console.warn('Could not load examples:', e);
        }
    },

    loadExample() {
        if (this.examples.length === 0) {
            // Fallback example
            this.examples = [{
                question: "What fraction of the shape is not shaded? [Image: A triangle split into 9 equal smaller triangles. 6 are shaded.]",
                answer: "1/3",
                explanation: "1/3 because 6 over 9 is 2 thirds and 1 third is not shaded."
            }];
        }
        const example = this.examples[Math.floor(Math.random() * this.examples.length)];
        document.getElementById('questionInput').value = example.question;
        document.getElementById('answerInput').value = example.answer;
        document.getElementById('explanationInput').value = example.explanation;
        document.getElementById('charCount').textContent = example.explanation.length;

        // Reset choice buttons
        document.querySelectorAll('.choice-btn').forEach(b => b.classList.remove('active'));

        Toast.show('Example loaded! Click "Analyze" to detect misconceptions.', 'success');
    },

    async analyze() {
        if (this.isLoading) return;

        const question = document.getElementById('questionInput').value.trim();
        const answer = document.getElementById('answerInput').value.trim();
        const explanation = document.getElementById('explanationInput').value.trim();

        if (!question || !answer || !explanation) {
            Toast.show('Please fill in all fields.', 'error');
            return;
        }

        this.isLoading = true;
        this.showLoading();

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, answer, explanation }),
            });

            if (!res.ok) throw new Error(`Server error: ${res.status}`);

            const data = await res.json();

            if (data.success) {
                this.showResults(data);
            } else {
                throw new Error(data.message || 'Prediction failed');
            }
        } catch (e) {
            Toast.show(`Error: ${e.message}`, 'error');
            this.showEmpty();
        } finally {
            this.isLoading = false;
            this.resetButton();
        }
    },

    showLoading() {
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'none';
        document.getElementById('loadingState').style.display = 'block';

        // Animate loading steps
        const steps = [
            'Running DeBERTa classifier...',
            'Computing BGE embeddings...',
            'Applying ensemble fusion...',
            'Running family filter...',
            'Generating explanation...',
        ];
        let i = 0;
        this._loadingInterval = setInterval(() => {
            i = (i + 1) % steps.length;
            const stepEl = document.getElementById('loadingStep');
            if (stepEl) stepEl.textContent = steps[i];
        }, 800);

        // Disable button
        const btn = document.getElementById('analyzeBtn');
        btn.disabled = true;
        btn.querySelector('span').style.display = 'none';
        btn.querySelector('.btn-loader').style.display = 'flex';
    },

    showEmpty() {
        clearInterval(this._loadingInterval);
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'none';
        document.getElementById('emptyState').style.display = 'block';
    },

    showResults(data) {
        clearInterval(this._loadingInterval);
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';

        // Reset feedback
        document.getElementById('feedbackThanks').style.display = 'none';
        document.querySelectorAll('.feedback-btn').forEach(b => b.style.display = 'inline-flex');

        const predictions = data.predictions;
        const top = predictions[0];

        // --- Top Prediction ---
        const catEl = document.getElementById('topCategory');
        catEl.textContent = top.category.replace('_', ' → ');
        catEl.style.background = `${top.category_color}20`;
        catEl.style.color = top.category_color;

        const misconEl = document.getElementById('topMisconception');
        misconEl.textContent = top.misconception === 'NA'
            ? 'No Specific Misconception'
            : top.misconception.replace(/_/g, ' ');

        document.getElementById('topDescription').textContent = top.category_description;

        // Confidence ring
        const conf = Math.round(top.confidence * 100);
        document.getElementById('topConfidenceText').textContent = `${conf}%`;
        const ringFill = document.getElementById('ringFill');
        const circumference = 2 * Math.PI * 45;
        ringFill.style.strokeDasharray = circumference;
        setTimeout(() => {
            ringFill.style.strokeDashoffset = circumference * (1 - top.confidence);
        }, 100);

        // --- Predictions List ---
        const listEl = document.getElementById('predictionsList');
        listEl.innerHTML = '';
        predictions.forEach((pred, idx) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <div class="prediction-rank rank-${pred.rank}">${pred.rank}</div>
                <div class="prediction-info">
                    <div class="prediction-label">${pred.label.replace(/_/g, ' ').replace(':', ' → ')}</div>
                    <div class="prediction-desc">${pred.misconception_description}</div>
                </div>
                <div class="prediction-bar-container">
                    <div class="prediction-conf">${(pred.confidence * 100).toFixed(1)}%</div>
                    <div class="prediction-bar">
                        <div class="prediction-bar-fill" style="width: 0"></div>
                    </div>
                </div>
            `;
            listEl.appendChild(item);

            // Animate bar
            setTimeout(() => {
                const bar = item.querySelector('.prediction-bar-fill');
                bar.style.width = `${Math.max(pred.confidence * 100, 5)}%`;
            }, 200 + idx * 150);
        });

        // --- Explanation ---
        const explEl = document.getElementById('explanationContent');
        // Convert markdown bold to HTML
        const formatted = data.explanation
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
        explEl.innerHTML = formatted;

        // Scroll to results
        document.getElementById('resultsContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    resetButton() {
        const btn = document.getElementById('analyzeBtn');
        btn.disabled = false;
        btn.querySelector('span').style.display = 'inline';
        btn.querySelector('.btn-loader').style.display = 'none';
    },

    async sendFeedback(helpful) {
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: Date.now().toString(),
                    helpful,
                    comment: null,
                }),
            });
            document.querySelectorAll('.feedback-btn').forEach(b => b.style.display = 'none');
            document.getElementById('feedbackThanks').style.display = 'flex';
            Toast.show('Thank you for your feedback!', 'success');
        } catch (e) {
            Toast.show('Could not send feedback.', 'error');
        }
    }
};

// ============================================================
// DASHBOARD
// ============================================================
const Dashboard = {
    loaded: false,

    async load() {
        if (this.loaded) return;

        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            this.renderTechniques(data.techniques || []);
            this.renderCategories(data.category_descriptions || {});
            this.loaded = true;
        } catch (e) {
            console.warn('Could not load dashboard data:', e);
            // Use fallback data
            this.renderTechniques([
                'Focal Loss (γ=2)',
                'Adversarial Weight Perturbation',
                'R-Drop Regularization',
                'Exponential Moving Average',
                'Layerwise LR Decay',
                'Temperature Scaling',
                'Family Filter',
                'GroupKFold Cross-Validation',
            ]);
            this.renderCategories({
                'True_Correct': 'Student answered correctly with proper reasoning',
                'True_Neither': 'Student answered correctly but explanation is unclear',
                'True_Misconception': 'Student answered correctly but reasoning has a misconception',
                'False_Correct': 'Student answered incorrectly but shows correct reasoning process',
                'False_Neither': 'Student answered incorrectly with unclear explanation',
                'False_Misconception': 'Student answered incorrectly due to a specific misconception',
            });
            this.loaded = true;
        }
    },

    renderTechniques(techniques) {
        const grid = document.getElementById('techniquesGrid');
        if (!grid) return;

        const icons = [
            'fa-crosshairs', 'fa-shield-halved', 'fa-droplet',
            'fa-chart-line', 'fa-layer-group', 'fa-temperature-three-quarters',
            'fa-filter', 'fa-sitemap'
        ];

        grid.innerHTML = techniques.map((t, i) => `
            <div class="technique-item">
                <div class="technique-icon">
                    <i class="fas ${icons[i % icons.length]}"></i>
                </div>
                <span>${t}</span>
            </div>
        `).join('');
    },

    renderCategories(categories) {
        const grid = document.getElementById('categoriesGrid');
        if (!grid) return;

        const colors = {
            'True_Correct': '#10b981',
            'True_Neither': '#f59e0b',
            'True_Misconception': '#f97316',
            'False_Correct': '#3b82f6',
            'False_Neither': '#8b5cf6',
            'False_Misconception': '#ef4444',
        };

        grid.innerHTML = Object.entries(categories).map(([name, desc]) => `
            <div class="category-item">
                <div class="category-dot" style="background: ${colors[name] || '#6b7280'}"></div>
                <div>
                    <div class="category-name">${name.replace('_', ' → ')}</div>
                    <div class="category-desc">${desc}</div>
                </div>
            </div>
        `).join('');
    }
};

// ============================================================
// INITIALIZE
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    Navigation.init();
    Analyzer.init();
});
