# Next Steps: Laptop And Kaggle

## Notebook Self-Contained Mode

`KAGGLE_RUN_FULL.ipynb` and `KAGGLE_RUN_CV.ipynb` now embed the current project code directly.

That means `CodeDs` is optional in Kaggle. The notebooks recreate the latest:

- `run_experiment.py`
- `src/`
- `configs/`
- `requirements.txt`

inside `/kaggle/working/embedded_codeds` before running anything.

## What I Ran Locally In This Workspace

These CPU-safe checks were run locally:

```powershell
python -m compileall src run_experiment.py
python -m src.baseline.qwen_adapter_infer --config configs/laptop_cpu_ds.yaml --prepare-only --split both --limit 16
python run_experiment.py --config configs/laptop_cpu_ds.yaml
python run_experiment.py --config configs/laptop_cpu_ds.yaml --run-cv
```

## What Those Local Runs Validate

These runs confirm:

- the repo can load `DS/train.csv` and `DS/test.csv`
- test rows get inferred `is_correct` and `family` from the train set
- the Qwen adapter bridge can prepare notebook-style classifier text without GPU
- the main OAMLLM-CV pipeline still runs end to end on CPU using the mock backend
- outputs, caches, graph files, and ablation files are written correctly

## What I Did Not Run Locally

I did not run the real Qwen3-8B adapter inference or the HF generative backend locally, because those are the GPU-heavy steps reserved for Kaggle H100.

## Kaggle Inputs To Attach

Attach these in Kaggle:

- `map_ds`
- `Base_models(H7b,q2.5-7b,3-8b)`
- `MAP_qwen3-8b_trained`

Useful exact paths from your current Kaggle setup:

- `/kaggle/input/datasets/mkhanshovon43/base-modelsh7bq2-5-7b3-8b/loaded_models/Qwen3-8B`
- `/kaggle/input/datasets/mkhanshovon43/base-modelsh7bq2-5-7b3-8b/loaded_models/Qwen2.5-7B-Instruct`
- `/kaggle/input/datasets/mkhanshovon43/map-qwen3-8b-trained/qwen3-8b-qlora/adapter`
- `/kaggle/input/datasets/mkhanshovon43/map-qwen3-8b-trained/qwen3-8b-qlora/label_encoder.joblib`
- `/kaggle/input/datasets/mkhanshovon43/map-qwen3-8b-trained/qwen3-8b-qlora/run_config.json`

## Recommended Kaggle Order

If you want the simplest path:

1. open `kaggle_notebooks/KAGGLE_RUN_FULL.ipynb`
2. run it from the top
3. after the full run works, optionally run `kaggle_notebooks/KAGGLE_RUN_CV.ipynb`

That is the preferred path now.

## What The Full Notebook Does

`KAGGLE_RUN_FULL.ipynb` now:

1. recreates the embedded code in `/kaggle/working/embedded_codeds`
2. auto-detects data and model paths
3. writes `/kaggle/working/runtime_kaggle_h100_qwen_full.yaml`
4. installs dependencies
5. exports baseline Qwen artifacts
6. builds KB and graph
7. runs the full ontology + counterfactual inference path

## Manual Kaggle Command Path

If you prefer shell commands after the notebook setup cell has run:

```bash
python -m src.baseline.qwen_adapter_infer --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --split both
python -m src.ontology.build_kb --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --output /kaggle/working/map_mvp/data/misconception_kb.json
python -m src.ontology.build_graph --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --mode full
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --run-cv
```

## Files To Inspect After Kaggle Runs

Look here first:

- `/kaggle/working/map_mvp/baseline_artifacts/`
- `/kaggle/working/map_mvp/output/`
- `/kaggle/working/map_mvp/graph/graph_summary.json`
- `/kaggle/working/map_mvp/logs/`

Especially:

- `qwen3_8b_adapter_test_scores_long.csv`
- `test_full_scores.csv`
- `test_full_counterfactual.csv`
- `test_full_submission.csv`
- `cv_metrics.csv`
- `ablation_summary.csv`

## If The Baseline Adapter Step Fails In Kaggle

The notebooks now print full stdout and stderr for each command.

Important updates already made:

- the notebooks prioritize `map_ds/DS/train.csv` and `map_ds/DS/test.csv`
- the notebooks now include your exact Kaggle dataset owner paths
- the adapter resolver can infer `adapter/` from the folder containing `label_encoder.joblib` or `run_config.json`
- the notebook code is embedded, so you do not need to refresh `CodeDs`

If it still fails:

1. rerun the notebook from the top so the embedded code refreshes in `/kaggle/working/embedded_codeds`
2. read the full stderr printed in the notebook output
3. send that stderr back directly
