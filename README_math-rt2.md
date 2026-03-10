# math-rt2.ipynb — MAP Pipeline (Best Pipeline)

> **Problem name:** MAP(Misconception Annotation Project) – Charting Student Math Misunderstandings
> **Task:** Given a student's multiple-choice math answer + written explanation, predict the **Category:Misconception** label (top-3 predictions evaluated with MAP@3).

---

## High-Level Idea

The notebook builds a **multi-stage NLP pipeline** that fine-tunes **DeBERTa-v3-Large** on structured student response data, hardens it with several regularisation techniques, and wraps the classifier with an **Explainable AI** layer that generates human-readable teacher feedback.

```
Raw CSV  →  EDA  →  Feature Engineering  →  DeBERTa Fine-Tuning  →  Post-Processing  →  Submission
                                                    ↑
                                         AWP + R-Drop + EMA + Focal Loss
                                         GroupKFold × 3 seeds (ensemble)
                                                    ↓
                                         Misconception KB + Teacher Feedback (Qwen 2.5 / FLAN-T5)
```

---

## Pipeline Stages

### 1. Exploratory Data Analysis
- 6 category types, 35 misconception types → **65 combined labels**
- Severe class imbalance (up to 14 800 : 1) → motivates Focal Loss
- Explanation length distribution → sets `MAX_LEN = 512`

### 2. Feature Engineering
- **Correct-answer inference:** for each question, the most common answer among students labelled `True_*` is taken as the ground-truth correct answer — without touching the test set.
- **Structured input template** fed to the tokeniser:
  ```
  Task: Classify the student's math response and identify any misconception.
  Question: <question text>
  Student's Answer: <MC answer>
  Correct Answer: <inferred>
  Is Student Correct: Yes / No / Unknown
  Student's Explanation: <free text>
  ```
- **Family masks** (`True_*` / `False_*`) precomputed for post-processing.

### 3. DeBERTa-v3-Large Fine-Tuning
| Setting | Value |
|---|---|
| Model | `microsoft/deberta-v3-large` |
| CV | GroupKFold by `QuestionId` (no leakage) |
| Seeds | 3 (42, 123, 456) → logit ensemble |
| Effective batch size | 32 (micro=16, grad accum=2) |
| Epochs | 3 |
| Learning rate | 1.5e-5 (layerwise decay 0.95) |
| Precision | BF16 |

**Key training techniques (each adds +0.3–2.0% MAP@3):**

| Technique | What it does |
|---|---|
| **Focal Loss** (γ=2) | Down-weights easy/frequent classes; focuses on rare misconceptions |
| **AWP** | Adversarially perturbs weights → flatter loss surface → better generalisation |
| **R-Drop** | Double forward pass per batch; minimises KL divergence between the two outputs (consistency regularisation) |
| **EMA** | Maintains exponential moving average of weights; used at eval/inference time |
| **Layerwise LR Decay** | Lower transformer layers get geometrically smaller LR; preserves pretrained knowledge |
| **Cosine Schedule + Warmup** | Smooth LR trajectory; avoids damaging pretrained weights at the start |

### 4. Post-Processing
- **Family Filter:** if `is_correct == 1`, all `False_*` logits → −∞ (and vice versa). Eliminates ~50% of the label space per sample.
- **Learned Temperature Scaling:** optimal temperature *T* is found by LBFGS minimisation of NLL on OOF logits, then applied as `logits / T` before softmax.

### 5. Explainable AI
- **Misconception Knowledge Base:** for each misconception type, prompt Qwen 2.5 7B (4-bit) or FLAN-T5-Large to generate a structured *knowledge card* (definition, wrong thought patterns, correct rule, fix steps, self-check question). Cached as `misconception_kb.json`.
- **Teacher Feedback Generator:** combines the DeBERTa prediction + KB card to produce personalised, step-by-step feedback for teachers, with rule-based and optional LLM-based verification.

### 6. Evaluation
- MAP@3 per fold, ensemble MAP@3, Top-1 / Top-3 accuracy
- Per-category accuracy breakdown
- Hardest-misconception analysis (lowest MAP@3 classes)
- Ablation study: cumulative impact of each technique (+~9% MAP@3 total)

### 7. Test Inference & Submission
- Loads all saved fold checkpoints → averages logits (proper fold ensemble)
- Applies Family Filter + learned temperature
- Writes `submission.csv` (`row_id`, `"Label1 Label2 Label3"`)

---

## Key Design Choices Summary

| Decision | Why |
|---|---|
| DeBERTa-v3-Large as backbone | Disentangled attention; SOTA on NLU benchmarks |
| GroupKFold (not random KFold) | Prevents same question appearing in train & val → honest score |
| Focal Loss over standard CE | 14 800:1 class imbalance makes standard CE train mostly on majority class |
| AWP + R-Drop together | Complementary: AWP targets weight-space flatness, R-Drop targets output consistency |
| Multi-seed ensemble | Reduces variance from random init; ~+0.5–1.0% MAP@3 for free |
| Explainability layer | Educational deployment requires teacher-facing explanations, not just labels |

---

## Output Files

| File | Description |
|---|---|
| `submission.csv` | Test predictions (top-3 labels per row) |
| `misconception_kb.json` | LLM-generated misconception knowledge cards |
| `fold_models/seed*_fold*/` | Saved DeBERTa checkpoints (one per seed × fold) |
| `results_summary.txt` | Key metrics summary (MAP@3, accuracy, config) |

---

## References

1. He et al. (2023) — DeBERTaV3. *ICLR 2023.* arXiv:2111.09543  
2. Wu et al. (2020) — AWP. *NeurIPS 2020.* arXiv:2004.05884  
3. Wu et al. (2021) — R-Drop. *NeurIPS 2021.* arXiv:2106.14448  
4. Lin et al. (2017) — Focal Loss. *ICCV 2017.* arXiv:1708.02002  
5. Guo et al. (2017) — Temperature Scaling. *ICML 2017.* arXiv:1706.04599  
6. Lakshminarayanan et al. (2017) — Deep Ensembles. *NeurIPS 2017.* arXiv:1612.01474  
7. Qwen Team (2024) — Qwen2.5. arXiv:2412.15115  
