# MAP Misconception Detection Deployment

This deployment folder contains the full-stack app:

- `frontend/` for the React UI
- `backend/` for the FastAPI API and model loading

This version is prepared to work with the notebook-aligned artifact flow from:

- `Extra\kaggle_notebooks\rf_inf_expAi.ipynb`
- `Extra\kaggle_notebooks\rf_results\map_mvp\`

## What is actually reused from `rf_results`

The deployable part of `rf_results` for this app is the richer misconception knowledge base:

- `rf_results\map_mvp\data\misconception_kb.json`

That KB now works directly with the backend explainer.

Important:

- `rf_results\map_mvp\output\*.csv` are preserved notebook outputs for Kaggle/reference.
- Those score files are not a standalone live model for brand new student answers.
- The live API still needs a deployed classifier adapter that matches its base model.

## Recommended backend setup

From `deployment/backend`, prepare data like this:

```powershell
python prepare_data.py --train-path ..\..\DS\train.csv --output-dir .\data --kb-path ..\..\Extra\kaggle_notebooks\rf_results\map_mvp\data\misconception_kb.json
```

This creates:

- `data/train.csv`
- `data/label_encoder.pkl`
- `data/correct_answers.json`
- `data/misconception_kb.json`

Then place your classifier adapter inside:

```text
deployment\backend\data\adapter\
```

For the safest Modal deployment, keep:

```text
USE_LLM_EXPLAINER=true
```

That keeps the deployed app using the full generative explainer path, assuming you also provide the matching adapter and model access.

## Environment variables

Backend `.env` values to check:

- `BASE_MODEL_PATH`
- `ADAPTER_PATH`
- `USE_4BIT_QUANTIZATION`
- `USE_LLM_EXPLAINER`
- `HF_TOKEN`

`BASE_MODEL_PATH` must match the adapter you deploy.

## Ordered commands

Use:

- `deployment/DEPLOYMENT_COMMANDS_IN_ORDER.md`

That file mirrors the earlier command-guide format, but now points the backend at the notebook-aligned `rf_results` knowledge base.
