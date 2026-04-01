# Run Commands

## CPU / Laptop

These are the commands I designed to be safe on a laptop CPU.
The first Qwen command only prepares notebook-style text previews and does not load any model.

```powershell
pip install -r requirements.txt
python -m compileall src run_experiment.py
python -m src.baseline.qwen_adapter_infer --config configs/laptop_cpu_ds.yaml --prepare-only --split both --limit 16
python run_experiment.py --config configs/laptop_cpu_ds.yaml
python run_experiment.py --config configs/laptop_cpu_ds.yaml --run-cv
```

## Kaggle H100

Run these in order.
The first command produces baseline artifacts from your trained Qwen3-8B adapter.
The later commands consume those artifacts inside the ontology/counterfactual pipeline.

```bash
pip install -r requirements.txt
python -m src.baseline.qwen_adapter_infer --config configs/kaggle_h100_qwen_full.yaml --split both
python -m src.ontology.build_kb --config configs/kaggle_h100_qwen_full.yaml --output /kaggle/working/map_mvp/data/misconception_kb.json
python -m src.ontology.build_graph --config configs/kaggle_h100_qwen_full.yaml
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode baseline
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode kb_only
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode counterfactual
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode full
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --run-cv
```

## What The Qwen Export Step Produces

`src.baseline.qwen_adapter_infer` writes:

- `baseline_artifacts_dir/qwen3_8b_adapter_train_scores_long.csv`
- `baseline_artifacts_dir/qwen3_8b_adapter_test_scores_long.csv`
- `baseline_artifacts_dir/qwen3_8b_adapter_train_topk.csv`
- `baseline_artifacts_dir/qwen3_8b_adapter_test_topk.csv`
- `output_dir/qwen3_8b_adapter_train_submission.csv`
- `output_dir/qwen3_8b_adapter_test_submission.csv`

Those `*_scores_long.csv` files are the ones the research pipeline reuses as the preserved baseline.

## Notes

- `configs/kaggle_h100_qwen_full.yaml` already points to common Kaggle path candidates for your `DS` dataset and Qwen adapter artifacts.
- If your uploaded Kaggle dataset names differ, update the path candidate lists in that config only.
- For the generative KB / counterfactual backend, upload one instruction-tuned model that matches one of `llm.model_candidates`, or edit that list.
- Read `NEXT_STEPS_LAPTOP_AND_KAGGLE.md` for the detailed runbook and what I already verified locally.
