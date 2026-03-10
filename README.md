# MAP – Charting Student Math Misunderstandings
### Machine Learning Project | CSE 330

---

## What Is This About?

This project solves a Kaggle NLP competition where the goal is to automatically detect **math misconceptions** in students' answers.

Given a math question, a student's chosen answer (A/B/C/D), and the student's written explanation, the model must predict **what misconception the student has** — ranked as a top-3 list. The evaluation metric is **MAP@3** (Mean Average Precision at 3).

$$\text{MAP@3} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{3} \sum_{k=1}^{3} P(k) \cdot \text{rel}(k)$$

Labels are combined strings like `True_Correct:NA` or `False_Misconception:Incomplete`, where the first part is the **Category** (whether the student was correct) and the second is the **Misconception type**.

## Dataset Description

Each row in `train.csv` / `test.csv` has these columns:

| Column | Description |
|--------|-------------|
| `row_id` | Unique row identifier |
| `QuestionId` | Which math question this row belongs to |
| `QuestionText` | The full text of the math question |
| `MC_Answer` | Which answer choice (A/B/C/D) the student selected |
| `StudentExplanation` | What the student wrote to explain their answer |
| `Category` | Target part 1 — was the student correct? (6 unique values) |
| `Misconception` | Target part 2 — what misconception? (35 unique values) |

There are **65 unique combined labels** (`Category:Misconception`). The dataset is **imbalanced** — about 40% of rows belong to `True_Correct:NA` (students who got it right with no misconception).

---

## How to Run

### Prerequisites

Install the required libraries:
```bash
pip install transformers==4.45.2 datasets==2.21.0 peft bitsandbytes accelerate
pip install torch scikit-learn pandas matplotlib seaborn
```

### Running on Kaggle

Change the paths at the top of the config cell:
```python
TRAIN_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
TEST_PATH  = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
```

### Running Locally

Update the paths in the config cell:
```python
TRAIN_PATH = "DS/train.csv"
TEST_PATH  = "DS/test.csv"
```
Then run cells in order.

### Important: Run cells top to bottom

Each cell depends on variables defined in earlier cells. Do not skip cells.

---

## Cell-by-Cell Walkthrough

---

### Cell 1 — Title & Architecture (Markdown)

This is a documentation cell. It shows:
- The full pipeline architecture as a text diagram
- The MAP@3 formula
- A table of all 13 academic references used in this pipeline

Nothing runs here — it is just for the reader/reviewer.

---

### Cell 2 — Environment Setup (Imports)

**What happens**: All libraries are imported and the GPU is detected.

Key libraries loaded:
- `transformers` — for DeBERTa model and tokenizer
- `datasets` — for fast data loading with HuggingFace Dataset objects
- `torch` — PyTorch deep learning framework
- `sklearn` — for GroupKFold cross-validation
- `matplotlib` / `seaborn` — for visualizations

The variable `device` is set to `"cuda"` if a GPU is available, otherwise `"cpu"`.

---

### Cell 3 — Configuration

**What happens**: All hyperparameters and settings are defined in one place.

Key settings:

| Variable | Value | Meaning |
|----------|-------|---------|
| `CLASSIFIER_MODEL` | `microsoft/deberta-v3-large` | The pre-trained model to fine-tune |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `SEEDS` | [42] | Single seed — deterministic, reproducible training |
| `EPOCHS` | 4 | Training epochs per fold |
| `BATCH_SIZE` | 4 | Samples per GPU step |
| `GRAD_ACCUM` | 8 | Accumulate 8 steps before updating (effective batch = 32) |
| `LR` | 1.5e-5 | Learning rate |
| `MAX_LEN` | 512 | Max token length for input text |
| `LABEL_SMOOTHING` | 0.1 | Softens the loss target by 10% |
| `AWP_LR` | 1e-4 | Step size for adversarial weight perturbation |
| `EMA_DECAY` | 0.999 | How fast the EMA model "forgets" old weights |
| `RDROP_ALPHA` | 0.5 | Weight of the consistency loss (R-Drop) |
| `TEMPERATURE` | 1.5 | Divides logits to soften probabilities |
| `DEBUG` | False | Set to True to do a quick 1-fold, 1-epoch test run |
| `OUTPUT_DIR` | `./output` | Directory where best fold checkpoints are saved |

---

### Cell 4 — Exploratory Data Analysis (EDA)

**What happens**: The training data is loaded and visualized.

The code:
1. Loads `train.csv` into a pandas DataFrame
2. Fills missing values (some students leave explanations blank)
3. Creates the combined label string `Category:Misconception`
4. Counts how often each label appears (class distribution)
5. Plots 4 charts:
   - Bar chart of the 20 most common labels
   - Pie chart of Category distribution
   - Bar chart of Misconception distribution
   - Box plot of explanation text length per Category

**Why this matters**: The heavy class imbalance (some misconceptions appear only ~10 times out of 36,696 rows) motivates the use of label smoothing and careful evaluation.

---

### Cell 5 — Feature Engineering

**What happens**: Raw data is transformed into model-ready features.

Three key functions are defined:

#### `build_correct_by_q(df)`
Figures out what the **correct answer** is for each question, by finding the row where `Category` starts with `True_Correct`. This is built **only from training data** to avoid leakage.

#### `compute_is_correct(df, correct_by_q)`
For each row in `df`, checks if the student's chosen answer matches the correct answer. Returns an array of 1s (correct) and 0s (incorrect).

#### `make_text(df, is_correct)`
Builds a single structured string for each row that the model will read. Format:
```
[CORRECT] Question: <question text> | Answer Choice: <MC_Answer> | Explanation: <StudentExplanation>
```
or
```
[INCORRECT] Question: <question text> | Answer Choice: <MC_Answer> | Explanation: <StudentExplanation>
```

The `[CORRECT]` / `[INCORRECT]` prefix is a strong signal — a correct student's misconception is almost always `NA`, while an incorrect student always has a real misconception.

Also, label encoding is done here:
- `classes` = sorted list of all 65 labels
- `lab2id` = maps label string → integer index
- `id2lab` = maps integer index → label string
- `true_mask` / `false_mask` = boolean arrays to know which label indices belong to correct/incorrect students

---

### Cell 6 — Advanced Training Components

**What happens**: First patches DeBERTa-v3's broken dropout, then defines six advanced training techniques as reusable classes and functions.

#### 0. StableDropout Patch *(FP16 NaN fix)*

DeBERTa-v3's internal `StableDropout` class uses `mask.to(x.dtype) / (1 - p)` inside attention layers. Under FP16 AMP (`autocast`), this mask cast overflows → produces NaN in the attention output → NaN logits → `Loss: nan`. This happens *before* your loss function is even called, which is why casting to float32 inside the loss function does not help.

**Fix**: The class is monkey-patched before any model is loaded, replacing it with standard `F.dropout` (which is numerically stable in FP16):

```python
import transformers.models.deberta_v2.modeling_deberta_v2 as _deberta_module
_deberta_module.StableDropout = _PatchedStableDropout  # uses F.dropout
```

This is the standard fix used by top Kaggle solutions for any DeBERTa-v3 + AMP setup.

#### 1. AWP (Adversarial Weight Perturbation)
*(Wu et al., NeurIPS 2020)*

Adds small calculated noise to the model weights during training to simulate harder inputs. This forces the model to learn more robust features. After computing the adversarial loss, the original weights are restored.

```
Normal forward → Loss → Backup weights → Perturb weights →
Adversarial forward → Adversarial loss → Restore weights → Combined update
```

#### 2. EMA (Exponential Moving Average)
*(Polyak & Juditsky, 1992)*

Maintains a "shadow" version of the model that is a running average of all past weight states. After training, the shadow model is used for inference instead of the last checkpoint — it is usually more stable and generalizes better.

#### 3. R-Drop (Regularized Dropout)
*(Wu et al., NeurIPS 2021)*

Runs the same input through the model **twice** (with different dropout masks), and adds a KL-divergence penalty to force both passes to produce similar predictions. This prevents overfitting from dropout randomness.

#### 4. LabelSmoothingCE
*(Szegedy et al., CVPR 2016)*

Instead of training the model to predict probability 1.0 for the correct class and 0.0 for all others, label smoothing assigns probability `1 - ε = 0.9` to the correct class and distributes `ε = 0.1` across all others. This prevents overconfidence.

#### 5. Layerwise LR Decay
*(Clark et al., ICLR 2020)*

Lower layers of a transformer (closer to the input) contain general language knowledge. Higher layers (closer to the output) contain task-specific knowledge. This function assigns **smaller learning rates** to lower layers so their pretrained knowledge is preserved, while upper layers adapt faster.

Decay per layer = `LR × decay_rate^(num_layers - layer_index)`

#### 6. Helper Functions

- **`apply_filters(logits, is_correct)`**: If a student is correct, zeroes out all logit scores for "incorrect" labels, and vice versa. This is called the "Family Filter" — it uses domain knowledge to eliminate impossible predictions.
- **`top3_from_logits(logits)`**: Selects the top-3 label indices after temperature scaling (`logits / TEMPERATURE`).
- **`map_at_3(true_labels, top3_preds)`**: Computes the MAP@3 metric score.

---

### Cell 6a — Pre-Training Inference Dry-Run *(NEW)*

**What happens**: Before spending hours training, this cell verifies that the **complete inference pipeline works end-to-end** on a few training samples.

**Why this is useful**: A training loop takes hours. If the inference code has a bug, you only discover it after all that time. This cell catches bugs early by doing the exact same inference steps — but on 6 sample rows from the training set.

**What "exact same path" means**:
The dry-run uses the **identical code steps** that will be used later on the test set:
- Same `CLASSIFIER_MODEL` (deberta-v3-large)
- Same `MAX_LEN` and tokenizer
- Same `Dataset.from_dict → tok_fn → DataLoader → forward pass`
- Same `apply_filters` and `top3_from_logits`
- Same `template_feedback` teacher feedback generator

**Why predictions are wrong on purpose**: The model weights are freshly downloaded from HuggingFace — the classification head has never been trained. So predictions will be random. This is **expected and normal**. The check is not whether predictions are correct, but whether the code runs without errors.

At the end, you will see:
```
✅ Dry-run PASSED — full inference path works correctly.
   Top-1 predictions are random (untrained head) — that is expected.
   Tokenizer + collator left in memory for the training cell.
```

**VRAM management**: The dry-run model is deleted from GPU memory at the end (`del dryrun_model`), but the **tokenizer and collator are kept in memory** so the training cell does not need to reload them.

---

### Cell 7 — DeBERTa Training Loop

**What happens**: The core model training using a single seed × GroupKFold cross-validation.

#### Why GroupKFold by QuestionId?

Multiple rows in the dataset share the same `QuestionId` (the same math question, but different students answering it). If we randomly split, the same question can appear in both train and validation — the model essentially memorizes the question and scores artificially high. 

**GroupKFold** ensures that all rows belonging to one `QuestionId` go to **either** train or validation — never both. This gives an honest performance estimate.

#### The Training Loop Structure

```
For seed = 42:
    For each fold in 1..5:
        ├── Split data by QuestionId (GroupKFold)
        ├── Build correct_by_q from TRAIN fold only
        ├── Compute is_correct for train + val
        ├── Build input texts with make_text()
        ├── Tokenize with HuggingFace Dataset
        ├── Load fresh DeBERTa-v3-large model
        ├── Enable gradient checkpointing (saves VRAM for R-Drop double forward)
        ├── Set up AdamW optimizer (with layerwise LR if enabled)
        ├── Set up cosine LR schedule with warmup
        └── For each epoch:
                ├── Forward pass #1
                ├── [R-Drop] Forward pass #2 + KL loss
                ├── Skip step if loss is NaN/Inf (NaN safety net)
                ├── Backward pass (gradient accumulation × 8)
                ├── [AWP] Adversarial step (after warm-up epoch)
                ├── Optimizer step + LR step
                ├── [EMA] Update shadow weights
                ├── [EMA] Apply shadow for validation
                ├── Compute val MAP@3
                ├── If best epoch → save to output/seed42_fold{N}_best.pt
                └── [EMA] Restore training weights
        └── Load best checkpoint for OOF logits:
                ├── Load seed42_fold{N}_best.pt
                ├── Run validation set through model
                ├── Store out-of-fold logits
                ├── Compute MAP@3 for this fold
                └── Delete model, free VRAM
```

OOF logits from all 5 folds are aggregated and used for evaluation in the next cell.

#### Checkpoint Saving

After **every epoch**, the model (with EMA weights applied) is evaluated on the validation fold. If the MAP@3 score is the best seen so far for that fold, the weights are saved:

```
output/
├── seed42_fold1_best.pt
├── seed42_fold2_best.pt
├── seed42_fold3_best.pt
├── seed42_fold4_best.pt
└── seed42_fold5_best.pt
```

Only the **best epoch** per fold is kept — there is no accumulation of per-epoch files. OOF logits are always computed from the best checkpoint (not the last epoch's weights).

#### Stability Fixes

**Gradient checkpointing** (`model.gradient_checkpointing_enable()`) is enabled after every model load. It trades GPU compute for memory by recomputing activations during the backward pass instead of storing them. This is required for DeBERTa-v3-large + R-Drop (double forward pass) on a T4 16 GB GPU without running out of VRAM.

**NaN safety net**: Even with the StableDropout patch, if any individual batch produces a NaN or Inf loss (e.g. from a degenerate input), the training step is skipped and the accumulated gradients are cleared instead of being applied. The epoch summary prints how many steps were skipped (`NaN skipped: N`). A healthy run shows 0 skipped steps.

> **Note**: Only model `state_dict` weights are saved. The tokenizer and label mappings (`lab2id` / `id2lab`) live in notebook memory. If you need fully standalone inference outside this notebook, you would also need to persist those (e.g., with `tokenizer.save_pretrained()` and `json.dump(lab2id, ...)`).  

---

### Cell 8 — Evaluation & Error Analysis

**What happens**: The out-of-fold predictions are analyzed with 3 visualizations:

1. **MAP@3 per fold** — a bar chart showing if any fold had unusually high/low performance (consistency check)
2. **Per-category accuracy** — how well the model predicts each of the 6 Category types
3. **Hardest misconceptions** — bar chart of the 10 misconception types with the most prediction errors

This cell helps identify where the model struggles (e.g., rare misconception types).

---

### Cell 9 — Explainable AI: Knowledge Base Builder

**What happens**: For each of the 35 misconception types, a structured **knowledge base (KB)** entry is generated using a Large Language Model.

#### The LLM Stack

Two LLMs are used, in priority order:

1. **`Qwen/Qwen2.5-7B-Instruct`** (primary) — a 7-billion parameter instruction-following model, loaded in **4-bit quantization** to fit on GPU. Very capable but requires a GPU.
2. **`google/flan-t5-large`** (fallback) — a smaller model that can run on CPU if Qwen cannot load.

#### What Each KB Entry Contains

For each misconception (e.g., `"Incomplete_Calculation"`), the LLM generates a JSON object:
```json
{
  "plain_definition": "The student stopped before completing all steps.",
  "common_triggers": ["multi-step problems", "arithmetic with fractions"],
  "fix_steps": [
    "Write out every step explicitly.",
    "Check if you have answered what the question asked.",
    "Verify your answer satisfies all conditions."
  ],
  "quick_check": "Did you answer every part of the question?"
}
```

The KB is saved to `misconception_kb.json` and reused if the file already exists (so the LLM is not called twice).

---

### Cell 10 — Teacher Feedback Generator

**What happens**: Using the KB from Cell 9, human-readable feedback is generated for each prediction.

#### Functions

- **`get_correct_choice(row, correct_by_q)`**: Looks up the correct answer for a given question.
- **`rule_based_verify(pred_label, is_correct)`**: Checks logical consistency (e.g., if student is correct, the predicted misconception must be `NA`).
- **`llm_verify(row, pred_label, llm_model)`**: Optionally asks the LLM to verify if the predicted misconception makes sense for this student's explanation.
- **`generate_teacher_feedback(row, top3_preds, kb, correct_by_q)`**: Combines KB lookup, rule checking, and LLM verification to produce a formatted feedback string.

A demo runs on 5 random training rows, printing feedback like:
```
Student chose  : B
Correct answer : A   (Is correct: No)
Predicted label: False_Misconception:Incomplete
Misconception  : The student stopped before completing all steps.

Fix steps:
  1. Write out every step explicitly.
  2. Check if you have answered what the question asked.

Self-check: Did you answer every part of the question?
```

---

### Cell 11 — Test Inference & Submission

**What happens**: All 5 saved fold checkpoints are loaded and ensembled to produce the final test predictions, then a `submission.csv` is generated.

**No retraining is done here.** The cell reads every `*.pt` file from `OUTPUT_DIR`, runs the test set through each fold model separately, and averages the raw logits — giving a proper 5-fold ensemble consistent with how OOF evaluation was done.

The inference pipeline:
1. Load `DS/test.csv`
2. Compute `is_correct` from the full training data's correct answers
3. Build input text with `make_text()`
4. Tokenize and load into a DataLoader
5. For each checkpoint in `output/seed42_fold{1..5}_best.pt`:
   - Load model weights with `model.load_state_dict(torch.load(ckpt))`
   - Forward pass through the model (AMP: `autocast`)
   - Collect logits
6. Average logits across all 5 fold models
7. `apply_filters` → remove impossible label families
8. `top3_from_logits` → temperature scaling + top-3 selection
9. Format as space-separated string: `"True_Correct:NA False_Neither:NA False_Misconception:Incomplete"`
10. Save to `submission.csv`

> If no checkpoints are found in `OUTPUT_DIR`, the cell raises a `RuntimeError` and asks you to run the training cell first.

Output format:
```
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
...
```

---

### Cell 12 — Ablation Study

**What happens**: Shows the expected contribution of each technique as a horizontal bar chart.

The table and chart show cumulative MAP@3 improvements as techniques are added one by one:

| Technique Added | Expected Gain |
|-----------------|:-------------:|
| GroupKFold (no leakage) | Honest baseline |
| Label Smoothing | +0.5–1.5% |
| Layerwise LR Decay | +0.5–1.0% |
| Cosine Schedule | +0.3–0.8% |
| R-Drop | +0.5–1.5% |
| AWP | +1.0–2.0% |
| EMA | +0.3–0.8% |
| Family Filter | +0.5–1.5% |
| Temperature Scaling | +0.2–0.5% |

> Total cumulative improvement: approximately **+3–7% MAP@3** over vanilla fine-tuning.

The chart uses placeholder values for visual demonstration — replace with actual numbers after running training.

---

### Cell 13 — Summary (Markdown)

A final summary table of all design choices and their justifications, followed by the complete 13-entry academic reference list.

---

## The Complete Pipeline at a Glance

```
Raw Data (train.csv)
       │
       ▼
Feature Engineering
  ├── Infer correct answer per QuestionId
  ├── Compute is_correct flag
  └── Build structured input text → "[CORRECT/INCORRECT] Question: ... | ..."
       │
       ▼
Pre-Training Dry-Run (verify inference path works)
       │
       ▼
Training (1 seed × 5 GroupKFolds)
  ├── StableDropout patch (FP16/AMP NaN fix for DeBERTa-v3)
  ├── DeBERTa-v3-large backbone + gradient checkpointing
  ├── AWP adversarial perturbation
  ├── R-Drop consistency regularization
  ├── EMA weight averaging
  ├── Label Smoothing loss
  ├── Layerwise LR Decay
  ├── Cosine LR Schedule with Warmup
  └── NaN safety net (skip corrupt batches)
       │
       ▼
Post-Processing
  ├── Family Filter (remove impossible label families)
  └── Temperature Scaling (soften probabilities)
       │
       ▼
Evaluation → MAP@3 per fold + error analysis charts
       │
       ▼
Explainable AI
  ├── LLM Knowledge Base (Qwen 7B → KB per misconception)
  └── Teacher Feedback Generator (KB + LLM verification)
       │
       ▼
Test Inference → submission.csv
```

---

## Common Questions

**Q: Why is the dry-run model making wrong predictions?**  
A: The model weights for the classifier head have just been initialized randomly — the model has not been trained yet. Wrong predictions at the dry-run stage are expected and normal. What we are checking is that the code runs without crashing.

**Q: What is `correct_by_q_full` vs `correct_by_q`?**  
A: `correct_by_q_full` is built from the entire training set and is used during the dry-run and test inference. `correct_by_q` (without `_full`) is rebuilt from only the training fold inside each cross-validation fold, to prevent data leakage.

**Q: Why was the loss `nan` during training?**  
A: DeBERTa-v3's internal `StableDropout` class casts its dropout mask to the input's dtype (`mask.to(x.dtype)`), then divides by `(1 - p)`. Under FP16 AMP, this overflows in the attention layers and produces NaN logits — before the loss function is even reached. The fix is to monkey-patch `StableDropout` with standard `F.dropout` (done in Cell 6, before any model is loaded). A NaN safety net in the training loop also skips any individual batch that still produces a NaN loss.

**Q: Why gradient accumulation?**  
A: The GPU cannot fit a batch of 32 samples at once. So we process 4 samples at a time (`BATCH_SIZE=4`) and accumulate gradients for 8 steps before updating the model. This gives the same result as using a batch of 32.

**Q: Why is gradient checkpointing enabled?**  
A: R-Drop runs two forward passes per batch, which nearly doubles the activation memory. DeBERTa-v3-large is already large (440M params). On a T4 16 GB GPU, this combination hits the VRAM limit without gradient checkpointing. With it enabled, activations are recomputed during the backward pass instead of stored — trading ~20% extra compute for a large VRAM saving.

**Q: How does the Family Filter work?**  
A: Labels are split into two "families": those starting with `True_*` (for correct students) and those starting with `False_*` (for incorrect students). Since we know whether the student is correct (`is_correct` flag), we can zero out all logits in the impossible family before selecting the top-3.

**Q: What does training on Kaggle look like?**  
A: Set `TRAIN_PATH` and `TEST_PATH` to the Kaggle paths, ensure a GPU accelerator is selected (P100 or T4), and `DEBUG=False`. Full training (1 seed × 5 folds × 4 epochs) takes approximately **2–3 hours** on Kaggle's free T4 GPU.

---

## References

1. He, P., Gao, J., & Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.* ICLR 2023. [arXiv:2111.09543](https://arxiv.org/abs/2111.09543)
2. Wu, D., Xia, S.-T., & Wang, Y. (2020). *Adversarial Weight Perturbation Helps Robust Generalization.* NeurIPS 2020. [arXiv:2004.05884](https://arxiv.org/abs/2004.05884)
3. Wu, X., et al. (2021). *R-Drop: Regularized Dropout for Neural Networks.* NeurIPS 2021. [arXiv:2106.14448](https://arxiv.org/abs/2106.14448)
4. Polyak, B. T. & Juditsky, A. B. (1992). *Acceleration of Stochastic Approximation by Averaging.* SIAM J. Control Optim.
5. Szegedy, C., et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
6. Clark, K., et al. (2020). *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.* ICLR 2020. [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)
7. Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
8. Guo, C., et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
9. Lakshminarayanan, B., et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
10. Qwen Team (2024). *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
11. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
12. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
13. Eedi & The Learning Agency (2025). *MAP – Charting Student Math Misunderstandings.* Kaggle. [Link](https://kaggle.com/competitions/map-charting-student-math-misunderstandings)
