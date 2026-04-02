# Run Commands

## Simplest Kaggle Path

If you are short on time, use the notebooks instead of typing commands manually.

Run these notebooks in Kaggle:

1. `kaggle_notebooks/KAGGLE_RUN_FULL.ipynb`
2. optional later: `kaggle_notebooks/KAGGLE_RUN_CV.ipynb`

The notebooks are now self-contained for project code, so `CodeDs` is optional.

## What To Attach In Kaggle

Attach these datasets/models:

- `map_ds`
- `Base_models(H7b,q2.5-7b,3-8b)`
- `MAP_qwen3-8b_trained`

Expected useful paths are:

- `/kaggle/input/datasets/mkhanshovon43/base-modelsh7bq2-5-7b3-8b/loaded_models/Qwen3-8B`
- `/kaggle/input/datasets/mkhanshovon43/base-modelsh7bq2-5-7b3-8b/loaded_models/Qwen2.5-7B-Instruct`
- `/kaggle/input/datasets/mkhanshovon43/map-qwen3-8b-trained/qwen3-8b-qlora/adapter`

## Manual CPU / Laptop Commands

These are safe on a laptop CPU. The Qwen command only prepares text previews and does not load the model.

```powershell
pip install -r requirements.txt
python -m compileall src run_experiment.py
python -m src.baseline.qwen_adapter_infer --config configs/laptop_cpu_ds.yaml --prepare-only --split both --limit 16
python run_experiment.py --config configs/laptop_cpu_ds.yaml
python run_experiment.py --config configs/laptop_cpu_ds.yaml --run-cv
```

## Manual Kaggle H100 Commands

Use these only if you want to run outside the notebooks.

```bash
pip install -r requirements.txt
python -m src.baseline.qwen_adapter_infer --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --split both
python -m src.ontology.build_kb --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --output /kaggle/working/map_mvp/data/misconception_kb.json
python -m src.ontology.build_graph --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --mode baseline
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --mode kb_only
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --mode counterfactual
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --mode full
python run_experiment.py --config /kaggle/working/runtime_kaggle_h100_qwen_full.yaml --run-cv
```

## What The Qwen Export Step Produces

`src.baseline.qwen_adapter_infer` writes:

- `baseline_artifacts_dir/qwen3_8b_adapter_train_scores_long.csv`
- `baseline_artifacts_dir/qwen3_8b_adapter_test_scores_long.csv`
- `baseline_artifacts_dir/qwen3_8b_adapter_train_topk.csv`
- `baseline_artifacts_dir/qwen3_8b_adapter_test_topk.csv`
- `output_dir/qwen3_8b_adapter_train_submission.csv`
- `output_dir/qwen3_8b_adapter_test_submission.csv`

Those `*_scores_long.csv` files are reused as the preserved baseline.

## Notes

- the notebooks now auto-build `/kaggle/working/runtime_kaggle_h100_qwen_full.yaml`
- the notebook setup includes exact fallback paths for your current Kaggle datasets
- the adapter loader can now infer `adapter/` from the folder containing `label_encoder.joblib` or `run_config.json`
- if the baseline export fails, rerun with the updated notebook because it now prints real stderr instead of only `CalledProcessError`
