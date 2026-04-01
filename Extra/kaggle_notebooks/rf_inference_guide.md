# RF Pipeline Inference Guide

## What `rf.ipynb` Does

`rf.ipynb` (despite its name) runs a **multi-stage misconception identification pipeline** for the Kaggle "MAP - Charting Student Math Misunderstandings" competition. It is NOT a simple Random Forest — it's a full ensemble pipeline with 5 major stages:

### Stage 1: Baseline Classification (Qwen3-8B + LoRA)
- Loads a **Qwen3-8B** language model fine-tuned with **QLoRA** (quantized Low-Rank Adaptation)
- Runs sequence classification on each test row
- Each row is formatted as:
  ```
  Question: <question text>
  Answer: <student's chosen answer>
  This answer is correct/incorrect.
  Student Explanation: <student's explanation>
  ```
- Produces softmax probabilities over all possible `Category:Misconception` labels
- Exports top-12 candidates per row with their probabilities

### Stage 2: Knowledge Base (KB) Construction
- Builds a structured knowledge base (`misconception_kb.json`) for each misconception
- Each KB entry contains:
  - `short_description`: What the misconception is
  - `prerequisite_concepts`: What math concepts are involved
  - `trigger_patterns`: Common n-gram patterns from student explanations that indicate this misconception
  - `formal_counterexample`: An example that disproves the misconception
  - `remediation_hint`: How to help a student overcome it
  - `concept_tags`: Tags like `decimal`, `fraction`, `algebra`, etc.
  - `cognitive_roots`: Deeper cognitive reasons for the misconception
  - `graph_links`: Related misconceptions

### Stage 3: Ontology Graph Construction
- Builds a NetworkX graph (`graph.gexf`) connecting related misconceptions
- Edges represent conceptual relationships (shared concepts, similar triggers, cognitive connections)
- Used for **GraphRAG** — retrieving related misconceptions as context for scoring

### Stage 4: KB/GraphRAG Scoring + Counterfactual Reasoning
- **KB Scoring**: Uses Qwen2.5-7B to score how well each candidate misconception matches the student's response, using KB context
- **GraphRAG**: Retrieves neighboring misconceptions from the graph to provide richer context for scoring
- **Counterfactual Reasoning**: For top candidates, generates a "what if the student didn't have this misconception?" rewrite of their explanation, then measures how much the explanation changes — bigger change = stronger evidence for the misconception
- Produces features: `kb_score`, `confidence`, `ontology_support_score`, `trigger_score`, `concept_score`, `subgraph_density`, `causal_delta`, `relative_delta`, `causal_confidence`, `style_preservation_score`

### Stage 5: Ensemble Fusion
- **Weighted fusion**: Combines all 14 features with hand-tuned weights:
  ```
  baseline_score:           0.18
  kb_score:                 0.15
  causal_delta:             0.14
  ontology_support_score:   0.12
  relative_delta:           0.08
  confidence:               0.07
  causal_confidence:        0.07
  trigger_score:            0.06
  concept_score:            0.05
  subgraph_density:         0.04
  style_preservation_score: 0.02
  graph_neighbor_count:     0.01
  trigger_match_count:      0.01
  ```
- **XGBoost reranker**: Trains a binary classifier on whether a candidate is the correct misconception, then blends:  
  `final_score = 0.7 * weighted_score + 0.3 * xgb_probability`
- **Family filtering**: Ensures correct-answer rows only get `True_*` predictions and incorrect-answer rows only get `False_*` predictions
- Takes top 3 predictions per row for the submission

---

## What the Outputs Mean

When you run `rf.ipynb` on Kaggle, it saves these files to `/kaggle/working/map_mvp/`:

### Baseline Artifacts (`baseline_artifacts/`)

| File | Description |
|------|-------------|
| `qwen3_8b_adapter_test_topk.csv` | Wide format: `row_id`, `family`, `label_1..label_12`, `prob_1..prob_12`, `prediction` — the top 12 predicted labels and their probabilities for each test row |
| `qwen3_8b_adapter_test_scores_long.csv` | Long format: `row_id`, `misconception_id`, `baseline_score` — one row per (test_row, candidate) pair with the baseline probability |
| `qwen3_8b_adapter_train_topk.csv` | Same as above but for training data (used for evaluation) |
| `qwen3_8b_adapter_train_scores_long.csv` | Same as above but for training data |

**How to read**: The `topk` files are human-readable — each row shows the top 12 predictions. The `scores_long` files are machine-readable — used by downstream pipeline stages.

### Knowledge Base (`data/`)

| File | Description |
|------|-------------|
| `misconception_kb.json` | A list/dict of structured misconception entries. Each entry has fields like `misconception_id`, `misconception_name`, `short_description`, `trigger_patterns`, `concept_tags`, etc. |

**How to read**: Open in any JSON viewer. Each entry describes one misconception with its properties, triggers, and remediation hints.

### Ontology Graph (`graph/`)

| File | Description |
|------|-------------|
| `graph.gexf` | NetworkX graph in GEXF format. Nodes = misconceptions, edges = conceptual relationships |
| `graph_summary.json` | Quick stats: number of nodes, edges, connected components |

**How to read**: Load with `networkx.read_gexf("graph.gexf")` or open in Gephi for visualization.

### Pipeline Output (`output/`)

| File | Description |
|------|-------------|
| `test_full_baseline_candidates.csv` | Initial candidate list from baseline classifier |
| `test_full_kb_scores.csv` | KB/GraphRAG scoring results: `kb_score`, `confidence`, `ontology_support_score`, `trigger_score`, `concept_score`, `subgraph_density`, `graph_neighbor_count`, `trigger_match_count`, `concept_match_count` |
| `test_full_counterfactual.csv` | Counterfactual reasoning results: `causal_delta`, `relative_delta`, `causal_confidence`, `style_preservation_score` |
| `test_full_scores.csv` | All features merged + `final_score` (the ensemble output) |
| `test_full_submission.csv` | **Final submission**: `row_id`, `prediction` (top 3 space-separated `Category:Misconception` labels) |

**How to read**: The `test_full_scores.csv` is the most informative — it shows all 14 features for each (row, candidate) pair plus the final fused score. Sort by `row_id` then `final_score` descending to see the ranking.

---

## How the Inference Notebook (`rf_inf_expAi.ipynb`) Uses These Outputs

The inference notebook has **two phases**:

### Phase 1: Score-Based Inference (Cells 1-9)

The notebook automatically detects which pre-computed artifacts are available and uses the best path:

| Available Artifacts | Path Used | What Happens |
|---|---|---|
| `test_full_scores.csv` (full scores) | **Path A** | Loads pre-computed ensemble scores, applies family filtering, takes top 3 → `submission.csv` |
| Only `test_scores_long.csv` (baseline) | **Path B** | Loads baseline probabilities, applies family filtering, takes top 3 → `submission.csv` |
| Nothing pre-computed | **Path C** | Loads Qwen3-8B + LoRA, runs live classification inference, applies family filtering → `submission.csv` |

### Phase 2: Explainable AI (Cells 10-13)

1. Frees GPU memory from any classification model
2. Loads Qwen3-8B as a **CausalLM** (text generation mode, not classification)
3. For a sample of training rows (with known ground truth), generates natural-language explanations:
   - Sends a prompt asking WHY a particular misconception was identified
   - The model explains what specific error in the student's reasoning reveals the misconception
4. Saves explanations to `explanations.json`
5. Prints a pipeline summary with prediction distribution stats

---

## How to Use the Inference Notebook

### Step 1: Save rf.ipynb Output as a Kaggle Dataset
1. Run `rf.ipynb` on Kaggle (you already did this)
2. Go to the notebook's output tab
3. Click "New Dataset" to save the output as a dataset
4. Name it something like `rf-output` or `rf-pipeline-output`

### Step 2: Create a New Kaggle Notebook
1. Create a new notebook on Kaggle
2. Upload `rf_inf_expAi.ipynb` or paste its contents

### Step 3: Attach Required Datasets
Add these datasets to your notebook:
- Your competition data (`map_ds`) — contains `train.csv`, `test.csv`, `misconception_mapping.csv`
- Qwen3-8B base model (e.g., `qwen-qwen3-8b` from Kaggle Models)
- Your trained LoRA adapter dataset (contains `adapter/` folder + `label_encoder.joblib`)
- The rf.ipynb output dataset you saved in Step 1

### Step 4: Update Paths (if needed)
In **Cell 3 (Configuration)**, update the candidate path lists if your dataset slugs differ from the defaults. The notebook tries multiple common path patterns automatically.

### Step 5: Run All Cells
- The notebook will auto-detect available artifacts and choose the optimal inference path
- Phase 1 produces `submission.csv`
- Phase 2 produces `explanations.json`

### Step 6: Submit
- Download `submission.csv` from the notebook output
- Submit to the Kaggle competition

---

## Key Concepts Explained

### What is Family Filtering?
Each prediction label follows the format `Category:Misconception` where `Category` starts with either `True_` (student answered correctly) or `False_` (student answered incorrectly). Family filtering ensures predictions match the row's correctness — a row where the student was wrong should only predict `False_*` misconceptions.

### What is the Label Encoder?
The `LabelEncoder` maps between human-readable labels like `"False_Misconception:ConfusesAreaAndPerimeter"` and numeric class indices (0, 1, 2, ...). The trained model outputs logits for each numeric class; the label encoder converts these back to readable labels.

### What is the Ensemble Doing?
Instead of relying on a single model's predictions, the pipeline combines 14 different signals:
- **Baseline score** (0.18 weight): Raw classifier probability — "how likely is this misconception based on the text?"
- **KB score** (0.15): LLM-judged match against structured misconception descriptions
- **Causal delta** (0.14): Counterfactual evidence — "how much does the explanation change if we remove this misconception?"
- **Ontology support** (0.12): Graph-based evidence from related misconceptions
- Other features provide additional signals from different angles

This multi-signal approach is more robust than any single model.

### What is MAP@3?
**Mean Average Precision at 3** — the competition metric. For each row, you submit 3 ranked predictions. You get full credit (1.0) if the correct answer is your #1 prediction, partial credit (0.5) if it's #2, less (0.33) if it's #3, and zero if it's not in your top 3. MAP@3 averages this across all rows.
