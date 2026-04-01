# How It Works

## Current Status

This repo now supports two connected paths:

1. your original-style Qwen baseline path through exported adapter inference artifacts
2. the research MVP path that reranks those baseline candidates with ontology context, counterfactual validation, and fusion

The important practical change after adding `DS/` and `Final_Model/` is that the code no longer assumes generic placeholder inputs. It now has a dedicated bridge for your Qwen3-8B QLoRA classifier workflow.

## Main Pipeline

1. `src/data/load_data.py`
   - Loads `DS/train.csv` and `DS/test.csv` from configurable path candidates.
   - Normalizes columns and now infers `is_correct` for test rows from the train set, matching your notebook logic.
   - Builds `family` so baseline family filtering works on test rows too.

2. `src/baseline/qwen_adapter_infer.py`
   - New GPU-oriented bridge for your `Final_Model` Qwen3-8B LoRA classifier.
   - Loads base model + adapter + optional label encoder.
   - Recreates the classifier input text format used in the training notebook.
   - Exports baseline artifacts into `baseline_artifacts_dir` in CSV formats that the research pipeline already knows how to read.

3. `src/baseline/pipeline.py`
   - Preserves the direct-comparison baseline path.
   - Reuses exported baseline artifacts first.
   - Falls back to a TF-IDF retriever only when those artifacts are missing.

4. `src/ontology/build_kb.py`
   - Builds and validates `data/misconception_kb.json`.
   - Keeps conservative ontology fields and merges existing curated content when present.

5. `src/ontology/graph_utils.py`
   - Builds the lightweight NetworkX graph.
   - Exports `graph.gexf` and `graph_summary.json`.

6. `src/graphrag/retrieve_context.py`
   - Pulls a bounded candidate-centered subgraph.
   - Turns it into a compact prompt context block.

7. `src/graphrag/score_candidates.py`
   - Scores top baseline candidates with strict JSON parsing.
   - Stores ontology and graph features for fusion.

8. `src/counterfactual/generate_counterfactual.py`
   - Rewrites the explanation as if the misconception were absent.
   - Keeps style-preservation metadata.

9. `src/counterfactual/score_counterfactual.py`
   - Rescores the rewritten answer.
   - Computes causal deltas and verdicts.

10. `src/ensemble/fuse_scores.py`
    - Fuses baseline, KB, graph, and counterfactual signals.
    - Optional XGBoost reranker remains available for CV / labeled settings.

11. `run_experiment.py`
    - Runs the main inference and submission path.
    - Supports `--mode baseline|kb_only|counterfactual|full` and `--run-cv`.

## New Practical Configs

- `configs/laptop_cpu_ds.yaml`
  - CPU-safe smoke-test config using the real `DS/` files with no large model.
- `configs/kaggle_h100_qwen_full.yaml`
  - Main Kaggle config for Qwen baseline export plus ontology/counterfactual reranking on H100.

## Research Alignment

The implementation still follows the same simplified research framing:

- baseline preservation from the 45th-place-style solution
- ontology / graph grounding for transparent retrieval
- GraphRAG-style context injection for candidate scoring
- counterfactual validation for causal confidence
- lightweight fusion for Kaggle practicality

## Important Simplifications

- local JSON + NetworkX instead of graph infrastructure
- bounded subgraphs instead of large GraphRAG communities
- exported baseline CSV artifacts instead of wiring notebook logic deep into the reranker
- strict JSON outputs with one repair pass
- weighted fusion first, XGBoost second

## Recommended Files To Open First

- `configs/kaggle_h100_qwen_full.yaml`
- `configs/laptop_cpu_ds.yaml`
- `src/baseline/qwen_adapter_infer.py`
- `src/experiment.py`
- `src/graphrag/score_candidates.py`
- `NEXT_STEPS_LAPTOP_AND_KAGGLE.md`
