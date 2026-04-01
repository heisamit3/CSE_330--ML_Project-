# Next Steps: Laptop And Kaggle

## What I Ran Locally In This Workspace

After you added `DS/` and `Final_Model/`, I updated the code and ran the CPU-safe path only.

Commands I ran:

```powershell
python -m compileall src run_experiment.py
python -m src.baseline.qwen_adapter_infer --config configs/laptop_cpu_ds.yaml --prepare-only --split both --limit 16
python run_experiment.py --config configs/laptop_cpu_ds.yaml
python run_experiment.py --config configs/laptop_cpu_ds.yaml --run-cv
```

## What Those Local Runs Validate

These local runs confirm:

- the repo can now load `DS/train.csv` and `DS/test.csv`
- test rows get inferred `is_correct` / `family` from the train set
- the new Qwen adapter bridge can prepare notebook-style classifier text without GPU
- the main OAMLLM-CV pipeline still runs end to end on CPU using the mock backend
- outputs, caches, graph files, and ablation files are still written correctly

## What I Did Not Run Locally

I did not run the actual Qwen3-8B adapter inference or the HF generative scoring backend locally because those are the GPU-heavy parts you explicitly want reserved for Kaggle H100.

## CPU-Safe Config

Use this for local smoke tests:

- `configs/laptop_cpu_ds.yaml`

It uses:

- the real `DS/` files in this repo
- no large model
- mock LLM backend
- small `max_rows` so it stays laptop-friendly

## Kaggle H100 Config

Use this for the actual GPU run:

- `configs/kaggle_h100_qwen_full.yaml`

This config expects two model families on Kaggle:

1. your trained Qwen3-8B classifier assets for the preserved baseline
2. one instruction-tuned generative model for KB-guided scoring and counterfactual rewriting

## How The Kaggle GPU Path Works

### Step 1. Upload or attach these Kaggle inputs

For the baseline classifier path:

- your `map-ds` dataset containing `DS/train.csv` and `DS/test.csv`
- the Qwen3-8B base model dataset
- the saved QLoRA output dataset containing:
  - `adapter/`
  - `label_encoder.joblib` if available
  - `run_config.json` if available

For the generative reasoning path:

- one instruction-tuned model matching one of the `llm.model_candidates` in `configs/kaggle_h100_qwen_full.yaml`

### Step 2. Export baseline artifacts from the trained Qwen adapter

Run:

```bash
python -m src.baseline.qwen_adapter_infer --config configs/kaggle_h100_qwen_full.yaml --split both
```

This produces the preserved baseline files inside:

- `/kaggle/working/map_mvp/baseline_artifacts/`

Those files are automatically consumed later by `run_experiment.py` because `baseline.use_external_artifacts` is already enabled.

### Step 3. Build ontology artifacts

Run:

```bash
python -m src.ontology.build_kb --config configs/kaggle_h100_qwen_full.yaml --output /kaggle/working/map_mvp/data/misconception_kb.json
python -m src.ontology.build_graph --config configs/kaggle_h100_qwen_full.yaml
```

### Step 4. Run the ablations

Run:

```bash
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode baseline
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode kb_only
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode counterfactual
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --mode full
python run_experiment.py --config configs/kaggle_h100_qwen_full.yaml --run-cv
```

## Important Path Details From Your Notebook

Your `Final_Model/Training/qwen3-8b.ipynb` saves:

- adapter: `/kaggle/working/qwen3-8b-qlora/adapter`
- label encoder: `/kaggle/working/qwen3-8b-qlora/label_encoder.joblib`
- run config: `/kaggle/working/qwen3-8b-qlora/run_config.json`

The new Kaggle config already includes those `working` paths plus common `/kaggle/input/...` candidates.
So if you either:

- rerun the training notebook in Kaggle first, or
- upload the saved QLoRA output as a Kaggle dataset

the new baseline export script can find it.

## If Something Needs A Small Edit On Kaggle

Only edit `configs/kaggle_h100_qwen_full.yaml`.
The most likely fields to adjust are:

- `paths.train_file`
- `paths.test_file`
- `baseline_qwen.base_model_candidates`
- `baseline_qwen.adapter_candidates`
- `baseline_qwen.label_encoder_candidates`
- `llm.model_candidates`

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

## What Changed In Code For Your Added Folders

The most relevant updates are:

- `src/data/load_data.py`
  - now supports path candidate lists and infers test correctness from train
- `src/baseline/qwen_adapter_infer.py`
  - new script that bridges `Final_Model` QLoRA outputs into baseline artifact CSVs
- `src/baseline/pipeline.py`
  - now respects `baseline.artifact_glob` so the exported Qwen files can be targeted directly
- `src/llm/backends.py`
  - now supports local model candidate resolution for offline Kaggle inputs
- `configs/laptop_cpu_ds.yaml`
  - real-data CPU smoke test config
- `configs/kaggle_h100_qwen_full.yaml`
  - main Kaggle GPU config tied to your dataset/model layout

## Short Recommendation

For the first Kaggle pass, do this exact order:

1. attach `map-ds`
2. attach or regenerate `qwen3-8b-qlora`
3. attach one instruct model for the generative backend
4. run `src.baseline.qwen_adapter_infer`
5. run `run_experiment.py --mode full`
6. then run `--run-cv`

That gives you the preserved old baseline plus the new ontology/counterfactual stack without mixing notebook logic directly into the reranker.


## If the baseline adapter cell fails in Kaggle

The Kaggle notebooks now print full stdout/stderr for each command instead of only showing `CalledProcessError`.

What changed:
- the notebooks now prioritize `map_ds/DS/train.csv` and `map_ds/DS/test.csv` before any generic `train.csv` or `test.csv`
- the runtime config detection is stricter for your MAP dataset and Qwen adapter layout
- the Qwen adapter loader now matches the training notebook more closely for sequence classification

What to do now:
1. Re-upload the updated `CodeDS` dataset if needed.
2. Re-run the setup cell in `KAGGLE_RUN_FULL.ipynb`.
3. Re-run the failing baseline export cell.
4. If it still fails, copy the full stderr that now appears in the notebook output.

