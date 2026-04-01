# Kaggle Notebook Instructions

This repo now has a silver-writeup-inspired two-notebook workflow for the MAP competition:

- `map_multi_llm_training.ipynb`
  Trains one selected model preset, validates it with family-aware MAP@3, optionally retrains on the full training set, and exports the full test-set probability matrix plus metadata.
- `map_multi_llm_submission.ipynb`
  Loads one or more exported run folders, ensembles their probability distributions, applies the `True_` / `False_` family filter, and writes the final `submission.csv`.

This is much closer to the linked 45th-place solution than the older single-model notebook, because the final submission is built from multiple model runs instead of one checkpoint.

## What The New Workflow Does

The linked writeup is built around three ideas:

- every model sees the same structured input with an `is_correct` signal
- each model outputs a full class-probability distribution
- the final answer comes from ensembling several models, then filtering predictions by family

The new notebooks follow that same pattern.

The practical difference from the old notebook is important:

- you train one preset at a time
- each run saves `test_probabilities.npz` instead of trying to ship one huge merged model
- the submission notebook combines those saved probability files into the final Kaggle submission

That makes the workflow lighter, easier to debug, and much more realistic for a multi-model ensemble.

## Files Created By The Training Notebook

Each training run writes to:

- `/kaggle/working/map_multi_llm_runs/artifacts/<run_slug>/`

Each run folder contains:

- `metadata.json`
- `validation_summary.json`
- `validation_predictions.csv`
- `test_probabilities.npz`
- `submission_preview.csv`

The most important file for the ensemble notebook is:

- `test_probabilities.npz`

That file stores the model's full probability distribution for every row in the competition test set.

## Recommended Kaggle Workflow

## 1. Open `map_multi_llm_training.ipynb`

Attach the competition dataset:

- `/kaggle/input/map-charting-student-math-misunderstandings`

Keep these paths unless you are running locally:

```python
TRAIN_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
TEST_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
```

## 2. Pick One Preset

Edit:

```python
ACTIVE_MODEL_KEY = "qwen3_4b_lora"
```

You can also change:

```python
SEED = 42
RUN_NAME_SUFFIX = ""
```

The notebook now saves each run as:

- `<preset>_seed<SEED>`

So you can train the same preset with multiple seeds without overwriting earlier runs.

Available presets in the notebook:

- `qwen3_4b_lora`
- `hunyuan_7b_lora`
- `qwen2_7b_qlora`
- `qwen3_14b_qlora`
- `qwen3_8b_qlora`
- `qwen3_8b_full`
- `deepseek_math_7b_full`

Notes:

- `qwen3_4b_lora` is the best place to start on Kaggle.
- `qwen2_7b_qlora` is the next most practical run.
- `qwen3_8b_qlora` is the closest preset to the standalone `Final_Model/Training/qwen3-8b.ipynb` workflow and is a good H100 option.
- `qwen3_14b_qlora`, `qwen3_8b_qlora`, `qwen3_8b_full`, and `deepseek_math_7b_full` may need significantly more VRAM than a free Kaggle GPU.
- The original writeup mentions a Qwen2 8B model; this notebook uses `Qwen/Qwen2-7B-Instruct` as the closest widely available public Qwen2 preset.

## 3. Run The Training Notebook

Run the notebook top to bottom.

What happens:

- the train data is preprocessed with the `is_correct` feature
- a leak-safe validation split is created
- the selected model is trained
- validation MAP@3 is reported with and without family filtering
- the model is optionally retrained on the full training set
- test-set probabilities are exported to the run folder

Useful toggles:

- `DEBUG = True` for a fast smoke test
- `DEBUG = False` for a real run
- `RUN_FINAL_FULL_TRAIN = True` for the normal export flow
- `SAVE_PROBABILITIES_AS_FLOAT16 = True` to keep the artifact dataset smaller

## 4. Repeat For More Models

Change only:

```python
ACTIVE_MODEL_KEY = "..."
```

Then rerun the notebook for another preset.

For higher accuracy, also repeat strong presets with different seeds, for example:

- `qwen3_4b_lora` with `SEED = 42`, `52`, `62`
- `qwen2_7b_qlora` with `SEED = 42`, `52`
- `hunyuan_7b_lora` with `SEED = 42` if it fits

Each run creates another folder under:

- `/kaggle/working/map_multi_llm_runs/artifacts/`

If you keep the same Kaggle session alive, the folder will accumulate multiple run subfolders.

## 5. Create One Kaggle Dataset From The Artifacts Folder

After you finish the model runs you want, create a Kaggle dataset from:

- `/kaggle/working/map_multi_llm_runs/artifacts`

The dataset root should contain folders like:

- `qwen3_4b_lora/`
- `qwen2_7b_qlora/`
- `hunyuan_7b_lora/`

## 6. Open `map_multi_llm_submission.ipynb`

Attach:

- the competition dataset
- the artifacts dataset you created in step 5

Then edit this cell:

```python
COMPETITION_DATA_DIR = Path("/kaggle/input/map-charting-student-math-misunderstandings")
ARTIFACT_ROOT = Path("/kaggle/input/map-multi-llm-artifacts")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
SELECT_RUN_SLUGS = None
TOP_N_BY_VALIDATION = None
MIN_VALIDATION_MAP3 = None
USE_VALIDATION_SCORE_WEIGHTS = True
VALIDATION_WEIGHT_POWER = 4.0
```

Important:

- `ARTIFACT_ROOT` must point to the dataset folder that contains the run subfolders directly
- leave `SELECT_RUN_SLUGS = None` to use every discovered run
- set `SELECT_RUN_SLUGS = ["qwen3_4b_lora", "qwen2_7b_qlora"]` if you only want a subset
- set `TOP_N_BY_VALIDATION = 3` if you want only the strongest few runs
- set `MIN_VALIDATION_MAP3 = 0.94` if you want to automatically drop weak runs
- leave `USE_VALIDATION_SCORE_WEIGHTS = True` so stronger validation runs influence the ensemble more

## 7. Run The Submission Notebook

The submission notebook will:

- discover all run folders
- load each saved probability matrix
- align rows and classes
- combine model probabilities with an agreement bonus and a confidence bonus
- apply the `True_` / `False_` family filter
- write `/kaggle/working/submission.csv`

## 8. Submit To Kaggle

The output format should be:

- one column only
- header: `Category:Misconception`
- exactly 3 labels per row separated by spaces

Example:

```csv
Category:Misconception
True_Neither:NA True_Correct:NA False_Neither:NA
```

## Practical Starting Point

If you want a realistic first attempt without trying to train the full six-model stack:

1. Run `qwen3_4b_lora` with `SEED = 42`
2. Run `qwen3_4b_lora` with `SEED = 52`
3. Run `qwen2_7b_qlora` with `SEED = 42`
4. If it fits, add `hunyuan_7b_lora` with `SEED = 42`
5. In the submission notebook, keep `USE_VALIDATION_SCORE_WEIGHTS = True`
6. Try `TOP_N_BY_VALIDATION = 3` and compare it against using all runs

That is usually a much stronger Kaggle setup than a single run, while still being far more feasible than the full silver-medal stack.

If you have an H100 and specifically want the stronger 8B route:

1. Set `ACTIVE_MODEL_KEY = "qwen3_8b_qlora"`
2. Keep `DEBUG = False`
3. Start with `SEED = 42`
4. After the run finishes, keep the exported folder under `/kaggle/working/map_multi_llm_runs/artifacts/`
5. Feed that artifact folder into `map_multi_llm_submission.ipynb` or the `Extra` predict config described below

## Local Paths

If you want to run the training notebook locally instead of in Kaggle, change:

```python
TRAIN_PATH = "DS/train.csv"
TEST_PATH = "DS/test.csv"
```

The local `DS/test.csv` is only a tiny smoke-test file. Use it to verify formatting, not to estimate leaderboard performance.
