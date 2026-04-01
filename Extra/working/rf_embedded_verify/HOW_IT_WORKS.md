# How It Works

## Current Flow

This project now supports two connected paths:

1. a preserved Qwen3-8B QLoRA baseline path
2. a research MVP reranking path with ontology context, counterfactual validation, and fusion

The important practical change is that the Kaggle notebooks are now self-contained. They embed the current project code and recreate it inside `/kaggle/working/embedded_codeds` before running anything.

That means:
- `CodeDs` is optional in Kaggle
- stale repo datasets no longer block you
- the notebook can run with only the data/model datasets attached

## Kaggle Inputs Expected

The current notebooks are written around your actual Kaggle layout:

- `map_ds`
  - `DS/train.csv`
  - `DS/test.csv`
  - optional `DS/misconception_mapping.csv`
- `Base_models(H7b,q2.5-7b,3-8b)`
  - `loaded_models/Qwen3-8B`
  - `loaded_models/Qwen2.5-7B-Instruct`
- `MAP_qwen3-8b_trained`
  - `qwen3-8b-qlora/adapter`
  - optional `qwen3-8b-qlora/label_encoder.joblib`
  - optional `qwen3-8b-qlora/run_config.json`

The notebooks also now include exact path fallbacks for your current Kaggle dataset owner path:
- `/kaggle/input/datasets/mkhanshovon43/base-modelsh7bq2-5-7b3-8b/loaded_models/Qwen3-8B`
- `/kaggle/input/datasets/mkhanshovon43/base-modelsh7bq2-5-7b3-8b/loaded_models/Qwen2.5-7B-Instruct`
- `/kaggle/input/datasets/mkhanshovon43/map-qwen3-8b-trained/qwen3-8b-qlora/adapter`

## Main Pipeline

1. `src/data/load_data.py`
- loads MAP CSVs from configurable path candidates
- normalizes columns
- infers `is_correct` and `family` for test rows from train when needed

2. `src/baseline/qwen_adapter_infer.py`
- loads the Qwen3-8B sequence-classification base model and LoRA adapter
- rebuilds the same classifier input format used in your training notebook
- exports baseline CSV artifacts used by the research pipeline
- now has stronger fallback logic for finding the adapter from `label_encoder.joblib` and `run_config.json`

3. `src/baseline/pipeline.py`
- preserves the old baseline comparison path
- reuses exported baseline artifacts first
- falls back to heuristic retrieval only if needed

4. `src/ontology/build_kb.py`
- builds and validates `misconception_kb.json`

5. `src/ontology/graph_utils.py`
- builds the lightweight NetworkX graph
- exports `graph.gexf` and `graph_summary.json`

6. `src/graphrag/retrieve_context.py`
- builds a compact candidate-centered context block from KB and graph neighbors

7. `src/graphrag/score_candidates.py`
- runs strict JSON scoring for candidate misconceptions
- stores graph and ontology support features

8. `src/counterfactual/generate_counterfactual.py`
- rewrites the explanation as if the misconception were absent

9. `src/counterfactual/score_counterfactual.py`
- rescoring step for the rewritten answer
- computes causal deltas and confidence signals

10. `src/ensemble/fuse_scores.py`
- fuses baseline, KB, graph, and counterfactual features
- optional XGBoost reranking remains available for CV

11. `run_experiment.py`
- main orchestration entrypoint
- supports `--mode baseline|kb_only|counterfactual|full` and `--run-cv`

## Kaggle Notebook Behavior

`kaggle_notebooks/KAGGLE_RUN_FULL.ipynb` now does this from the top:

1. recreates the embedded project code in `/kaggle/working/embedded_codeds`
2. auto-detects your Kaggle data/model paths
3. writes `/kaggle/working/runtime_kaggle_h100_qwen_full.yaml`
4. installs dependencies
5. runs Qwen baseline export
6. builds KB and graph
7. runs the full OAMLLM-CV inference path

`kaggle_notebooks/KAGGLE_RUN_CV.ipynb` does the same setup, then runs CV and ablations.

## Recommended Files To Open First

- `kaggle_notebooks/KAGGLE_RUN_FULL.ipynb`
- `kaggle_notebooks/KAGGLE_RUN_CV.ipynb`
- `configs/kaggle_h100_qwen_full.yaml`
- `src/baseline/qwen_adapter_infer.py`
- `NEXT_STEPS_LAPTOP_AND_KAGGLE.md`
