# Deployment Commands in Order

This file turns `deployment/README.md` into an ordered command checklist.

It is written for PowerShell on Windows.

## Important note before you start

This command order follows the deployment README, but for this repo's current backend code the safest setup is:

- keep the LoRA adapter inside `deployment/backend/data/adapter/`
- set `ADAPTER_PATH=./data/adapter` in the backend `.env`

That matches how the current classifier code loads adapters.

## Part 1: Prepare the Backend

### Step 1

**Work needed before executing this command**

- Make sure you are in the project root folder.
- Make sure Python 3.11+ is installed.

**Command**

```powershell
cd 'c:\D drive\L3-T2\CSE330 Machine Learning Sessional\Project\CSE_330--ML_Project-\deployment\backend'
```

**What will be done after executing it**

- Your terminal will move into the backend deployment folder.
- All next backend commands will run from the correct location.

### Step 2

**Work needed before executing this command**

- You should already be inside `deployment/backend`.

**Command**

```powershell
python -m venv venv
```

**What will be done after executing it**

- A virtual environment named `venv` will be created.
- This gives you an isolated Python environment for backend dependencies.

### Step 3

**Work needed before executing this command**

- The `venv` folder should already exist from the previous step.

**Command**

```powershell
.\venv\Scripts\Activate.ps1
```

**What will be done after executing it**

- The virtual environment will become active in your current terminal.
- Any `pip install` commands after this will go into the project environment.

### Step 4

**Work needed before executing this command**

- The virtual environment should be active.
- Internet access is needed so `pip` can download packages.

**Command**

```powershell
pip install -r requirements.txt
```

**What will be done after executing it**

- FastAPI, PyTorch, Transformers, PEFT, and other backend dependencies will be installed.
- The backend code will become runnable in this environment.

### Step 5

**Work needed before executing this command**

- The virtual environment should still be active.
- The dataset file `DS/train.csv` should exist in the project.

**Command**

```powershell
python prepare_data.py --train-path ..\..\DS\train.csv --output-dir .\data
```

**What will be done after executing it**

- Backend data artifacts will be generated inside `deployment/backend/data`.
- This will create:
  - `train.csv`
  - `label_encoder.pkl`
  - `correct_answers.json`
  - `misconception_kb.json`

### Step 6

**Work needed before executing this command**

- You should have your trained LoRA adapter files available somewhere on your machine.
- If you do not have the adapter yet, stop here because real model deployment is not complete without it.

**Command**

```powershell
New-Item -ItemType Directory -Force .\data\adapter
```

**What will be done after executing it**

- The adapter folder will be created if it does not already exist.
- This prepares the expected location for your trained adapter files.

### Step 7

**Work needed before executing this command**

- Replace `C:\path\to\your\adapter\*` with the real location of your adapter files.
- Your adapter folder should contain files such as `adapter_config.json` and adapter weights.

**Command**

```powershell
Copy-Item 'C:\path\to\your\adapter\*' '.\data\adapter' -Recurse -Force
```

**What will be done after executing it**

- Your trained LoRA adapter files will be copied into `deployment/backend/data/adapter`.
- The backend will then be able to load the adapter from the local deployment folder.

### Step 8

**Work needed before executing this command**

- You should still be inside `deployment/backend`.

**Command**

```powershell
Copy-Item .env.example .env
```

**What will be done after executing it**

- A backend `.env` file will be created from the example template.
- You will then be ready to add your real environment values.

### Step 9

**Work needed before executing this command**

- Open `deployment/backend/.env` in your editor.
- Update at least these values before going further:
  - `HF_TOKEN=your_real_huggingface_token`
  - `ADAPTER_PATH=./data/adapter`
- Save the file after editing.

**Command**

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**What will be done after executing it**

- The backend will start locally on port `8000`.
- FastAPI will try to load the prepared data files and the model.
- You can use this as a local smoke test before deploying to Modal.

### Step 10

**Work needed before executing this command**

- Keep the backend server running from the previous step.
- Open a second terminal if needed.

**Command**

```powershell
Invoke-RestMethod http://localhost:8000/health
```

**What will be done after executing it**

- You will get the health response from the backend.
- This helps confirm whether the API and model loaded successfully.

## Part 2: Deploy the Backend to Modal

### Step 11

**Work needed before executing this command**

- The backend folder should already be prepared.
- Your Python environment can stay active.
- Internet access is required.

**Command**

```powershell
pip install modal
```

**What will be done after executing it**

- The Modal CLI and Python package will be installed.
- Your machine will be ready to deploy the backend to Modal.

### Step 12

**Work needed before executing this command**

- You need a Modal account.
- Be ready to complete the authentication prompt in the browser or terminal.

**Command**

```powershell
modal setup
```

**What will be done after executing it**

- Your local machine will be authenticated with Modal.
- Future `modal` commands will be able to deploy and manage this backend.

### Step 13

**Work needed before executing this command**

- You need your real Hugging Face token.
- Replace `your_huggingface_token_here` with the real token value.

**Command**

```powershell
modal secret create hf-secret HF_TOKEN=your_huggingface_token_here
```

**What will be done after executing it**

- A Modal secret named `hf-secret` will be created.
- The deployed backend will be able to access Hugging Face models using that token.

### Step 14

**Work needed before executing this command**

- Make sure you are still in `deployment/backend`.
- Make sure `data/` contains the generated files and your adapter folder.
- Make sure the Modal secret has already been created.

**Command**

```powershell
modal deploy modal_app.py
```

**What will be done after executing it**

- Modal will build the backend image, upload the app, and deploy the API.
- You will receive Modal endpoint URLs such as:
  - health
  - predict
  - submit
  - question

### Step 15

**Work needed before executing this command**

- Copy the real health URL printed by Modal after deployment.
- Replace the sample URL below with your actual deployed health endpoint.

**Command**

```powershell
Invoke-RestMethod https://your-username--map-misconception-detector-misconceptionapi-health.modal.run
```

**What will be done after executing it**

- You will test the deployed backend directly on Modal.
- This confirms whether the cloud deployment is alive and serving requests.

## Part 3: Prepare and Deploy the Frontend

### Step 16

**Work needed before executing this command**

- Node.js 18+ should be installed.
- You can use a new terminal, or stop the local backend terminal if you no longer need it.

**Command**

```powershell
cd '..\frontend'
```

**What will be done after executing it**

- Your terminal will move into the frontend deployment folder.
- The next commands will install and configure the React app.

### Step 17

**Work needed before executing this command**

- You should already be inside `deployment/frontend`.
- Internet access is required.

**Command**

```powershell
npm install
```

**What will be done after executing it**

- The React frontend dependencies will be installed.
- The frontend will be ready to run locally or deploy to Vercel.

### Step 18

**Work needed before executing this command**

- You should still be inside `deployment/frontend`.

**Command**

```powershell
Copy-Item .env.example .env.local
```

**What will be done after executing it**

- A local frontend environment file will be created.
- You will then be able to point the frontend to the correct backend URL.

### Step 19

**Work needed before executing this command**

- Open `deployment/frontend/.env.local` in your editor.
- For local backend testing, set:
  - `REACT_APP_API_BASE_URL=http://localhost:8000`
- For deployed frontend use, set it to the backend URL pattern you want the frontend to call.
- Save the file before continuing.

**Command**

```powershell
npm start
```

**What will be done after executing it**

- The React development server will start locally.
- You will be able to open the frontend in the browser and test the full flow.

### Step 20

**Work needed before executing this command**

- Only do this if you want to deploy the frontend to Vercel.
- Make sure the frontend has already been tested locally.
- Internet access is required.

**Command**

```powershell
npm install -g vercel
```

**What will be done after executing it**

- The Vercel CLI will be installed globally.
- Your machine will be ready to deploy the frontend.

### Step 21

**Work needed before executing this command**

- Configure the frontend environment in Vercel so it points to your deployed backend.
- Be ready to answer the Vercel CLI prompts.

**Command**

```powershell
vercel --prod
```

**What will be done after executing it**

- The frontend will be built and deployed to Vercel.
- You will get a public frontend URL that users can open in the browser.

## Final Order Summary

If you only want the short order list, it is:

1. `cd deployment/backend`
2. `python -m venv venv`
3. `.\venv\Scripts\Activate.ps1`
4. `pip install -r requirements.txt`
5. `python prepare_data.py --train-path ..\..\DS\train.csv --output-dir .\data`
6. `New-Item -ItemType Directory -Force .\data\adapter`
7. `Copy-Item 'C:\path\to\your\adapter\*' '.\data\adapter' -Recurse -Force`
8. `Copy-Item .env.example .env` then edit `.env`
9. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
10. `Invoke-RestMethod http://localhost:8000/health`
11. `pip install modal`
12. `modal setup`
13. `modal secret create hf-secret HF_TOKEN=...`
14. `modal deploy modal_app.py`
15. test the Modal health URL
16. `cd ..\frontend`
17. `npm install`
18. `Copy-Item .env.example .env.local` then edit `.env.local`
19. `npm start`
20. `npm install -g vercel`
21. `vercel --prod`
