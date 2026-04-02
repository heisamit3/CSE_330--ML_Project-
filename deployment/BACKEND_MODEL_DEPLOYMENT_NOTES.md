# Backend and Model Deployment Notes

This file summarizes what I checked and concluded from the deployment setup.

## What I reviewed

- `deployment/README.md`
- `deployment/backend/app/config.py`
- `deployment/backend/app/main.py`
- `deployment/backend/app/models/classifier.py`
- `deployment/backend/app/models/explainer.py`
- `deployment/backend/app/services/prediction.py`
- `deployment/backend/modal_app.py`
- `deployment/frontend/src/services/api.js`
- `deployment/backend/.env.example`
- `deployment/backend/requirements.txt`

## What I found

### 1. The intended deployment flow is valid

The project is set up for:

- Frontend on Vercel
- Backend and model on Modal
- FastAPI for the API
- Qwen3-8B based classifier and explainer

### 2. Backend data artifacts must be prepared first

Run this from `deployment/backend`:

```powershell
python prepare_data.py --train-path ..\..\DS\train.csv --output-dir .\data
```

This generates:

- `data/train.csv`
- `data/label_encoder.pkl`
- `data/correct_answers.json`
- `data/misconception_kb.json`

### 3. The LoRA adapter is the main missing piece

I did not find adapter files in the repo.

Expected location:

```text
deployment/backend/data/adapter/
```

Expected files include:

- `adapter_config.json`
- adapter weight file such as `adapter_model.safetensors`

### 4. Important mismatch with the README

The README suggests you can set `ADAPTER_PATH` to a Hugging Face Hub repo ID.

But the current backend code in `deployment/backend/app/models/classifier.py` only loads the adapter if `ADAPTER_PATH` exists as a local folder and contains `adapter_config.json`.

So with the current code:

- local adapter folder works
- Hugging Face repo ID will not load as a LoRA adapter
- if adapter is missing, the app falls back to the base model only

### 5. Frontend Modal support is already implemented

The README says you may need to update `deployment/frontend/src/services/api.js`.

I checked it, and it already contains Modal URL handling logic.

So if you use the correct Modal-style base URL, you probably do not need to change that file.

### 6. Modal deployment expects mounted local data

`deployment/backend/modal_app.py` mounts:

- `./data` to `/app/data`
- `./app` to `/app/app`

It also expects a Modal secret named:

```text
hf-secret
```

That secret must contain:

```text
HF_TOKEN=your_huggingface_token
```

## Recommended deployment steps

### 1. Prepare backend environment

```powershell
cd 'c:\D drive\L3-T2\CSE330 Machine Learning Sessional\Project\CSE_330--ML_Project-\deployment\backend'
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py --train-path ..\..\DS\train.csv --output-dir .\data
```

### 2. Add your trained LoRA adapter

Put the adapter files inside:

```text
deployment\backend\data\adapter\
```

### 3. Configure backend env

Create `.env` from `.env.example` and set at least:

```text
HF_TOKEN=your_huggingface_token
ADAPTER_PATH=./data/adapter
```

### 4. Optional local backend test

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Invoke-RestMethod http://localhost:8000/health
```

If local model loading is too heavy, you can temporarily set:

```text
USE_LLM_EXPLAINER=true
```

### 5. Deploy backend to Modal

```powershell
pip install modal
modal setup
modal secret create hf-secret HF_TOKEN=your_huggingface_token
modal deploy modal_app.py
```

### 6. Test deployed endpoints

Modal will give separate endpoint URLs, for example:

- `...-health.modal.run`
- `...-predict.modal.run`
- `...-submit.modal.run`

Test health with the exact URL Modal prints.

## Final conclusion

The backend/model deployment structure is mostly ready.

The main blocker for a real deployment is the missing LoRA adapter folder.

Without that adapter, the backend will run, but it will use only the base model instead of your trained adapter.
