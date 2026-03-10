# Conformal Prediction for Reliable Student Misconception Detection

> **Notebook**: `conformal_prediction_misconception.ipynb`  
> **Base Pipeline**: `math_Hacc_modify.ipynb` (DeBERTa-v3-large + BGE-Large Ensemble)  
> **Status**: Ready to run after base pipeline generates OOF data

---

## Overview

This notebook implements **Conformal Prediction** on top of the existing DeBERTa + BGE ensemble pipeline to provide **mathematically guaranteed reliable predictions** for student math misconception detection.

Instead of giving just a single top-k prediction (which can be confidently wrong), conformal prediction outputs a **prediction set** with a coverage guarantee:

> *"The true misconception is in this set at least 95% of the time — this is a mathematical guarantee, not a heuristic."*

### Why This Matters for Education

| Scenario | Without Conformal | With Conformal |
|----------|-------------------|----------------|
| Easy misconception | "Misconception A" (correct ✅) | "Misconception A" (set size 1) → 🟢 Trust |
| Ambiguous case | "Misconception B" (wrong ❌) | "B, C, or D" (set size 3) → 🟡 Review |
| Very rare case | "Misconception E" (wrong ❌) | "E, F, G, H, I" (set size 5) → 🟠 Careful |

The system is **honest about uncertainty** — it says "I don't know" instead of being confidently wrong.

---

## What's Implemented

### Three Conformal Prediction Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| **THR** | Threshold-based (simplest) — includes all classes above a probability threshold | Vovk et al., 2005 |
| **APS** | Adaptive Prediction Sets — accumulates classes by descending probability until quantile is met | Romano, Sesia & Candès, NeurIPS 2020 |
| **RAPS** | Regularized APS (**recommended**) — adds a penalty for large sets, producing the tightest predictions | Angelopoulos et al., ICLR 2021 |

### Analyses & Visualizations

1. **Coverage vs. Set Size Trade-off** — Pareto analysis across 5 alpha levels (80%–99% coverage)
2. **Set Size Distribution** — How many predictions are "trust" (size 1), "review" (size 2-3), or "expert needed" (size 6+)
3. **Per-Class Analysis (Rare vs. Common)** — Shows rare misconceptions get larger sets (more honest), common ones get smaller sets
4. **Confusability Analysis** — Entropy-based difficulty vs. set size correlation
5. **Teacher Decision Support Interface** — Practical categorization into actionable groups
6. **Monte Carlo Validation** — 100 random splits confirm the coverage guarantee is robust
7. **LaTeX-ready Tables** — Copy-paste for research papers

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     CONFORMAL PREDICTION LAYER                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  INPUT: Softmax probabilities from DeBERTa + BGE Ensemble         │
│    └─ α × DeBERTa_probs + (1-α) × Retrieval_probs                │
│                                                                    │
│  STEP 1: CALIBRATION (on OOF data)                                │
│    ├─ Compute nonconformity scores for each calibration sample     │
│    │   • THR: s = 1 - p(y_true)                                   │
│    │   • APS: s = cumulative prob until true class                 │
│    │   • RAPS: s = APS_score + λ·max(0, rank - k_reg)             │
│    └─ Find quantile q̂ at level ⌈(n+1)(1-α)⌉/n                   │
│                                                                    │
│  STEP 2: PREDICTION (on test data)                                │
│    ├─ For each sample, include classes until score ≥ q̂            │
│    └─ Guarantee: P(Y ∈ C(X)) ≥ 1 - α                             │
│                                                                    │
│  STEP 3: TEACHER DECISION SUPPORT                                 │
│    ├─ Size 1   → 🟢 Trust blindly                                 │
│    ├─ Size 2-3 → 🟡 Quick review                                  │
│    ├─ Size 4-5 → 🟠 Careful review                                │
│    └─ Size 6+  → 🔴 Expert needed                                 │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## How to Run

### Prerequisites

1. **Run the base pipeline first** (`math_Hacc_modify.ipynb`) — this trains the DeBERTa + BGE ensemble and generates OOF predictions
2. The base pipeline now automatically saves OOF data at the end of Cell 8:
   - `oof_deberta_logits.npy` — DeBERTa out-of-fold logits
   - `oof_retrieval_probs.npy` — BGE retrieval out-of-fold probabilities
   - `oof_labels.npy` — True label indices

### Step-by-Step

```bash
# 1. Run base pipeline (math_Hacc_modify.ipynb)
#    This generates the OOF data files

# 2. Run conformal prediction notebook
#    Open conformal_prediction_misconception.ipynb and run all cells

# 3. Output files:
#    - conformal_prediction_results.json (all metrics in JSON)
#    - conformal_coverage_vs_setsize.png
#    - conformal_set_size_distribution.png
#    - conformal_per_class_analysis.png
#    - conformal_difficulty_analysis.png
#    - conformal_teacher_interface.png
#    - conformal_monte_carlo_validation.png
```

### Demo Mode (No GPU Required)

If OOF data files are not available, the notebook **automatically generates simulated probabilities** for demonstration purposes. This lets you explore the conformal prediction framework immediately, but the results will not reflect real model performance. You'll see a warning message directing you to run the base pipeline first for real results.

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRIMARY_ALPHA` | 0.05 | Main error tolerance (95% coverage guarantee) |
| `ALPHA_LEVELS` | [0.01, 0.05, 0.10, 0.15, 0.20] | Coverage levels to analyze |
| `RAPS_LAMBDA` | 0.01 | RAPS regularization strength (penalizes large sets) |
| `RAPS_K_REG` | 5 | Start penalizing after k-th element in sorted probs |
| `ENSEMBLE_ALPHA` | 0.65 | DeBERTa weight in ensemble (update from base pipeline) |
| `LEARNED_TEMPERATURE` | 1.5 | Temperature scaling (update from base pipeline) |

---

## Mathematical Guarantee

The coverage guarantee of split conformal prediction is:

$$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \geq 1 - \alpha$$

where:
- $Y_{n+1}$ is the true label for a new test sample
- $\mathcal{C}(X_{n+1})$ is the prediction set for that sample
- $\alpha$ is the user-specified error tolerance (e.g., $\alpha = 0.05$ for 95% coverage)

This guarantee is **distribution-free** — it holds regardless of:
- The true data distribution
- The model being miscalibrated
- Class imbalance (up to 740:1 in our dataset)

The only assumption is **exchangeability** (weaker than i.i.d.) of calibration and test data.

---

## Expected Results

| Method | Coverage (target 95%) | Avg Set Size | Singleton % |
|--------|:---------------------:|:------------:|:-----------:|
| THR    | ≥ 0.950               | Larger       | Lower       |
| APS    | ≥ 0.950               | Medium       | Medium      |
| RAPS   | ≥ 0.950               | **Smallest** | **Highest** |

**RAPS** is recommended for deployment because it produces the smallest prediction sets while maintaining the coverage guarantee.

### Per-Class Behavior

- **Common misconceptions** (e.g., `True_Correct:NA`): Average set size ~1-2 → teacher can trust
- **Medium misconceptions**: Average set size ~3-4 → teacher should review options
- **Rare misconceptions** (bottom 25% by frequency): Average set size ~5-8 → model is honestly uncertain

---

## Publication Potential

### Novelty

1. **First application** of conformal prediction to educational misconception detection
2. **Combined with domain-specific features**: family filter (True_*/False_*), temperature scaling, ensemble calibration
3. **Per-class analysis** revealing how conformal sets interact with severe class imbalance
4. **Practical teacher interface** translating mathematical guarantees into actionable decisions

### Target Venues

- **Tier 1 ML**: NeurIPS, ICLR (conformal prediction is trending — 10+ papers per venue in 2024-2025)
- **Education AI**: AIED, LAK, EDM (trustworthy education AI is highly valued)
- **Applied ML**: AAAI, IJCAI

### Potential Paper Titles

- *"Conformal Prediction Sets for Reliable Student Misconception Detection"*
- *"Uncertainty-Aware Math Misconception Detection with Distribution-Free Coverage Guarantees"*
- *"How Confident Should We Be? Conformal Prediction for Trustworthy Educational AI"*

---

## References

1. Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
2. Romano, Y., Sesia, M. & Candès, E. (2020). *Classification with Valid Adaptive Prediction Sets.* NeurIPS 2020. [arXiv:2006.02544](https://arxiv.org/abs/2006.02544)
3. Angelopoulos, A.N. et al. (2021). *Uncertainty Sets for Image Classifiers using Conformal Prediction.* ICLR 2021. [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)
4. Angelopoulos, A.N. et al. (2022). *Conformal Risk Control.* [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
5. Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
6. He, P. et al. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.* ICLR 2023. [arXiv:2111.09543](https://arxiv.org/abs/2111.09543)

---

## File Structure

```
├── conformal_prediction_misconception.ipynb   ← This notebook
├── math_Hacc_modify.ipynb                     ← Base pipeline (generates OOF data)
├── README_conformal_prediction.md             ← This README
├── DS/
│   ├── train.csv
│   └── test.csv
│
│  (Generated by base pipeline)
├── oof_deberta_logits.npy
├── oof_retrieval_probs.npy
├── oof_labels.npy
│
│  (Generated by this notebook)
├── conformal_prediction_results.json
├── conformal_coverage_vs_setsize.png
├── conformal_set_size_distribution.png
├── conformal_per_class_analysis.png
├── conformal_difficulty_analysis.png
├── conformal_teacher_interface.png
└── conformal_monte_carlo_validation.png
```
