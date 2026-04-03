# Code Evolution — How the Pipeline Developed

*Chronological walkthrough of every notebook, why each step was taken, and what measurable improvement it brought.*

---

## Overview

| # | Notebook | Approach | Score |
|---|----------|----------|-------|
| 1 | `1. deberta(val = 65).ipynb` | DeBERTa-v3-Large baseline fine-tuning | MAP@3 ≈ 0.65 |
| 2 | `2. math-acc-full-train(val = 80).ipynb` | DeBERTa + suffix classification + multi-seed ensemble | MAP@3 ≈ 0.80 |
| 3 | `3. map-qwen3-8b(training).ipynb` | Qwen3-8B QLoRA fine-tuning with advanced regularization | Training notebook |
| 4 | `4. q3-8b_infer(score=.93999).ipynb` | Qwen3-8B adapter inference (simple, single-stage) | 0.93999 |
| 5 | `rf.ipynb` | Full pipeline: KB + Graph + GraphRAG + Counterfactual + Ensemble | Full pipeline run |
| 6 | `rf_inf_expAi(score=.93999).ipynb` | Final: Classification + rf.ipynb artifacts + Explainable AI | **0.93999** |

---

## Stage 1 — DeBERTa Baseline
**File:** `Previous codes/1. deberta(val = 65).ipynb`
**Score:** MAP@3 ≈ 0.65

### What was done
- Fine-tuned **DeBERTa-v3-Large** (435M parameters) as a multi-class classifier over all misconception labels.
- Used 5-fold `GroupKFold` cross-validation (groups = question ID) to prevent data leakage across folds.
- Applied **Exponential Moving Average (EMA)** over model weights for smoother, more stable inference.
- Ensemble: averaged logits across all folds, then applied **temperature scaling** to calibrate confidence.

### Why this was a reasonable starting point
DeBERTa-v3 was the best encoder-only model available for classification tasks at the time. GroupKFold ensured the evaluation reflected realistic out-of-distribution performance since the same question never appears in both train and validation.

### Limitation identified
The model saw misconception labels as plain integers. It had no structural knowledge of *what* one misconception label means relative to another — the 2,000+ class head was essentially a lookup table. Top-1 accuracy was only ~54%, meaning the correct answer was often rank 2 or 3, not rank 1. MAP@3 was 0.65.

---

## Stage 2 — Suffix Classification + Multi-Seed Ensemble
**File:** `Previous codes/2. math-acc-full-train(val = 80).ipynb`
**Score:** MAP@3 ≈ 0.80 (+15 points over Stage 1)

### What was done
- Kept DeBERTa-v3-Large but changed the **label representation**: instead of raw integer IDs, the target string became `"Category:MisconceptionText"` (e.g., `"Fractions:Divides numerator and denominator separately"`). This is called **suffix classification**.
- This forces the model to learn semantic similarity between misconception names, not arbitrary numeric IDs.
- Ran **multiple seeds × 5 folds** and ensembled all checkpoints — more diverse predictions, lower variance.
- Added a **knowledge base** (generated with Qwen2.5-7B-Instruct) that explains each misconception; used to build richer input context.
- **Temperature grid search** was done on MAP@3 directly (not cross-entropy), making calibration task-aligned.

### Why this improved over Stage 1
The suffix representation gives the model semantic signal. DeBERTa's attention mechanism can now exploit the words in the misconception label itself — "Fractions" and "Divides" are meaningful tokens. This alone accounted for most of the +15 MAP@3 gain.

### Remaining limitation
DeBERTa is a 435M parameter encoder. It cannot reason — it pattern-matches. For ambiguous or novel student explanations, it still confuses misconceptions that sound similar. The next step was to use a much larger, generative model.

---

## Stage 3 — Qwen3-8B QLoRA Training
**File:** `Previous codes/3. map-qwen3-8b(training).ipynb`

### What was done
Switched to **Qwen3-8B** (8 billion parameters, ~18× larger than DeBERTa) and fine-tuned it with:

| Technique | Purpose | Reported gain |
|-----------|---------|--------------|
| **QLoRA** (4-bit NF4 + LoRA r=16, α=32) | Fit an 8B model into Kaggle's 16GB VRAM without full fine-tuning | Enables training at all |
| **R-Drop** (α=0.5) | Two forward passes per step; KL-divergence loss forces dropout consistency | +0.5–1.5% MAP |
| **AWP** (lr=0.1, ε=1e-4, starts epoch 1) | Perturbs weights slightly, computes loss again, restores — flattens loss surface | +1.0–2.0% MAP |
| **EMA** (decay=0.999) | Shadow copy of weights; used at eval and final save — more stable than last checkpoint | +0.3–0.8% MAP |
| **Label smoothing** | Prevents overconfident predictions on 2000-class head | Regularisation |
| **Layerwise LR decay** (0.95× per layer) | Deeper layers (pre-trained representations) get smaller LR; top layers adapt faster | Better fine-tuning |
| **AdamW + gradient clipping** (max_norm=1.0) | Standard LLM fine-tuning with decoupled weight decay | Stability |

### Why a larger model
Qwen3-8B is a decoder-only Transformer with deep reasoning capacity. It was loaded as `AutoModelForSequenceClassification`, so the full attention stack processes the student explanation before the classification head predicts the misconception. The much larger capacity allows it to capture nuance that DeBERTa missed.

### Role of this notebook in the rest of the pipeline
This notebook's output — the saved **LoRA adapter** (`adapter_model.safetensors`) and the **LabelEncoder** (`label_encoder.joblib`) — are the artifacts that all downstream notebooks depend on.

---

## Stage 4 — Simple Qwen3-8B Inference
**File:** `Previous codes/4. q3-8b_infer(score=.93999).ipynb`
**Score:** 0.93999

### What was done
- Loaded the Qwen3-8B base + the LoRA adapter trained in Stage 3.
- Ran direct classification inference on the test set.
- Produced `submission.csv` with top-12 misconception candidates per row.

### Why this was a big jump
Pure inference from the 8B model immediately achieved **0.93999**, up from 0.80 (DeBERTa ensemble). The larger model's capacity and the advanced training tricks (AWP, R-Drop, EMA) paid off.

### Remaining limitation
This is a single-model, single-stage pipeline. There is no external knowledge, no re-ranking, and no way to explain *why* a prediction was made. If the model is wrong, there is no mechanism to catch or correct it. The next stage adds all of that.

---

## Stage 5 — Full Pipeline with GraphRAG + Counterfactual + Ensemble
**File:** `rf.ipynb` → outputs stored in `rf_results/`

### What was done
`rf.ipynb` orchestrates a **four-stage pipeline** on top of the Stage 3 trained model:

```
Stage 1  →  Qwen3-8B adapter inference (top-12 candidates per row)
Stage 2  →  Knowledge Base construction (misconception_kb.json)
Stage 3  →  Ontology graph construction (NetworkX DiGraph, 7 edge types)
Stage 4  →  Full OAMLLM-CV inference:
              4a. GraphRAG retrieval (KB entry + up to 3 graph-neighbor misconceptions)
              4b. LLM scoring (Qwen2.5-7B-Instruct scores each candidate given KB context)
              4c. Counterfactual generation (rewrite student explanation as if misconception absent)
              4d. Ensemble fusion (14 features weighted → XGBoost meta-learner)
```

**14 features in the ensemble fusion:**

| Feature | Weight | Source |
|---------|--------|--------|
| `baseline_score` | 0.18 | Qwen3-8B classification probability |
| `kb_score` | 0.15 | LLM-scored affinity to KB entry |
| `ontology_support_score` | 0.12 | Graph path support (RoG-inspired) |
| `causal_delta` | 0.14 | Confidence drop when misconception removed (counterfactual) |
| `relative_delta` | 0.08 | Relative causal delta |
| `subgraph_density`, `graph_neighbor_count`, ... | distributed | Graph structure features |

### Why this is better than Stage 4
Stage 4 is a black box. Stage 5 provides:
- **Evidence**: KB retrieval shows *what* the model knows about the misconception.
- **Causal verification**: Counterfactuals confirm whether the misconception actually *caused* the student's phrasing.
- **Graph-aware re-ranking**: Related misconceptions that share cognitive roots are considered together.
- **Robustness**: Even if the 8B model is slightly wrong, the 14-feature ensemble can correct the ranking.

Results from this stage are saved to `rf_results/` and used by the final notebook.

---

## Stage 6 — Final Notebook: Inference + Explainable AI
**File:** `rf_inf_expAi(score=.93999).ipynb`
**Score:** 0.93999

### What was done
This is the production notebook. It runs in two phases:

**Phase 1 — Classification**
- Loads Qwen3-8B + LoRA adapter (from Stage 3).
- Checks for pre-computed `rf.ipynb` score artifacts. If available, uses the full 14-feature ensemble scores; otherwise falls back to live Qwen3-8B inference.
- Applies **family filtering**: filters candidates to the correct misconception category family first, improving precision.
- Produces `submission.csv`.

**Phase 2 — Explainable AI**
- Frees GPU memory from the classification model.
- Reloads Qwen3-8B as a **CausalLM** (text generator, not classifier).
- For each sampled prediction, generates a natural-language explanation:
  > *"The student's answer suggests misconception X because they wrote '...'. The key phrase '...' matches the trigger pattern '...' in the knowledge base."*
- Saves all explanations to `explanations.json`.

### Why this is the best version

| Problem in earlier stages | How Stage 6 solves it |
|--------------------------|----------------------|
| DeBERTa: no semantic label knowledge | Qwen3-8B processes label text via attention |
| Stage 4: single model, no re-ranking | GraphRAG + counterfactuals provide external evidence |
| Stage 4: opaque predictions | Explainable AI generates human-readable justifications |
| Stage 5: no family pre-filtering | Family filtering in Stage 6 reduces false positives |
| Stages 1–4: no interpretability for teacher/student | `explanations.json` shows *why* the model flagged the misconception |

---

## Full Dependency Chain

```
3. map-qwen3-8b(training).ipynb
         │
         │  saves: adapter_model.safetensors
         │          label_encoder.joblib
         │
         ▼
      rf.ipynb
         │  uses: Qwen3-8B adapter
         │  builds: KB → Graph → GraphRAG → Counterfactual → Ensemble
         │  saves: rf_results/ (full scored candidates)
         │
         ▼
rf_inf_expAi(score=.93999).ipynb
         │  Phase 1: classification (uses rf_results if present)
         │  Phase 2: Explainable AI (CausalLM explanations)
         │
         ▼
    submission.csv  +  explanations.json
```

---

## Score Progression Summary

```
Stage 1 (DeBERTa baseline)           MAP@3 ≈ 0.65
Stage 2 (DeBERTa + suffix labels)    MAP@3 ≈ 0.80   [+0.15]
Stage 3 (Qwen3-8B training)          —  (training only)
Stage 4 (Qwen3-8B simple inference)  0.93999         [+0.14]
Stage 5 (rf.ipynb full pipeline)     —  (intermediate artifacts)
Stage 6 (rf_inf_expAi final)         0.93999         (+ explainability)
```

The largest single gain (+0.14) came from switching from DeBERTa to a fine-tuned 8B-parameter model trained with QLoRA, AWP, R-Drop, and EMA. The pipeline around it (GraphRAG, counterfactuals, ensemble) adds robustness and interpretability without sacrificing the score.
