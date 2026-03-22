# How the `deployment/` Folder Works

This file explains the purpose of the `deployment/` folder, how its parts connect, and what happens when the app runs.

## Big Picture

The `deployment/` folder is a deployment-ready version of the project split into two parts:

- `deployment/backend/` for the API and ML model
- `deployment/frontend/` for the student-facing web app

The idea is:

1. The frontend shows a question and collects the student's answer and explanation.
2. The frontend sends that data to the backend API.
3. The backend loads the question data, runs the classifier, generates feedback, and sends a response back.
4. The frontend displays the verdict, misconception details, and teacher-style feedback.

## Top-Level Structure

### `deployment/README.md`

This is the deployment guide. It explains the intended hosting setup:

- frontend on Vercel
- backend on Modal

### `deployment/backend/`

This contains everything needed to run the API and the model:

- app code
- model loading logic
- data preparation script
- Modal deployment file
- Dockerfile
- environment template

### `deployment/frontend/`

This contains the React app that talks to the backend:

- UI components
- hooks for loading questions and submitting answers
- API service code
- Vercel config

## How the Backend Works

### Main backend purpose

The backend is a FastAPI service that:

- loads question data
- loads the classifier and explainer
- exposes endpoints like `/health`, `/predict`, and `/api/submit`

### Important backend files

#### `deployment/backend/prepare_data.py`

This script creates the backend data artifacts from `DS/train.csv`.

It generates:

- `data/train.csv`
- `data/label_encoder.pkl`
- `data/correct_answers.json`
- `data/misconception_kb.json`

These files are important because the backend depends on them during startup.

#### `deployment/backend/app/config.py`

This file reads environment variables such as:

- base model path
- adapter path
- data file paths
- Hugging Face token
- quantization options

It acts as the central settings file for the backend.

#### `deployment/backend/app/main.py`

This is the FastAPI entry point.

On startup it:

1. loads the question bank from `train.csv`
2. loads the prediction service
3. makes the API ready

It also defines the HTTP routes:

- `GET /health`
- `GET /labels`
- `POST /predict`
- `GET /api/question`
- `GET /api/questions/list`
- `POST /api/submit`
- `POST /api/rate`

#### `deployment/backend/app/services/prediction.py`

This is the main orchestration layer.

It combines:

- the classifier
- the explainer
- correct answer lookup
- rule-based postprocessing

When a prediction request comes in, this service:

1. infers whether the answer is correct if needed
2. formats the input
3. runs classification
4. filters and ranks predictions
5. generates teacher feedback
6. builds the final response object for the frontend

#### `deployment/backend/app/models/classifier.py`

This loads the Qwen3-4B sequence classification model.

Its job is to predict labels like:

- correct vs incorrect pattern
- misconception category

Important detail:

The classifier currently loads a LoRA adapter only from a local folder path.

That means the expected setup is:

```text
deployment/backend/data/adapter/
```

If that folder is missing, the backend falls back to the base model only.

#### `deployment/backend/app/models/explainer.py`

This creates feedback for the student.

It uses:

- the same Qwen base model in CausalLM mode when available
- the misconception knowledge base JSON as support

If the explainer model cannot load, it can still fall back to KB-based feedback.

### Backend data flow

The backend depends on the `deployment/backend/data/` folder.

That folder is meant to contain:

- generated training-derived artifacts
- optionally the LoRA adapter

So the normal flow is:

1. run `prepare_data.py`
2. populate `data/`
3. start FastAPI or deploy to Modal

## How the Model Part Works

There are really two model-related pieces in the backend.

### 1. Classifier

The classifier predicts the student's misconception-related label from:

- question text
- selected answer
- student explanation
- correctness

This is the main prediction engine.

### 2. Explainer

The explainer turns the prediction into readable feedback such as:

- what the student seems to think
- what misconception was detected
- how to fix the reasoning
- a self-check prompt

So the backend is not only classifying; it is also producing instructional feedback.

## How Modal Deployment Works

### `deployment/backend/modal_app.py`

This file adapts the backend for Modal.

It does a few specific things:

1. builds a Python image with ML dependencies
2. mounts local `./app` into the container
3. mounts local `./data` into the container
4. loads the prediction service during container startup
5. exposes Modal web endpoints

The mounted paths inside Modal are:

- `/app/app`
- `/app/data`

This is why the local `deployment/backend/data/` folder matters so much.

### Modal endpoint style

In local FastAPI mode, endpoints look like:

- `http://localhost:8000/health`
- `http://localhost:8000/api/question`

In Modal mode, endpoints are split into separate URLs such as:

- `...-health.modal.run`
- `...-predict.modal.run`
- `...-submit.modal.run`
- `...-question.modal.run`

That difference is already handled in the frontend API service.

## How the Frontend Works

### Main frontend purpose

The frontend is a React app for student practice.

It:

- shows a question
- lets the student choose an answer
- collects an explanation
- sends the answer to the backend
- shows feedback and misconception information

### Important frontend files

#### `deployment/frontend/src/App.js`

This is the main UI controller.

It manages:

- theme
- session start
- question number
- toast messages
- question loading
- answer submission
- feedback display

#### `deployment/frontend/src/hooks/useQuestion.js`

This hook loads a question from the backend.

It calls the API service and stores:

- the question
- loading state
- error state

#### `deployment/frontend/src/hooks/useSubmit.js`

This hook sends the student's answer and explanation to the backend.

It stores:

- feedback
- loading state
- error state

#### `deployment/frontend/src/services/api.js`

This is the connection layer between frontend and backend.

It knows how to call:

- local FastAPI URLs
- Modal-style URLs

So the frontend can work in both local development and deployed mode.

#### `deployment/frontend/vercel.json`

This config is for deploying the React app to Vercel.

It defines:

- the build command
- the output folder
- SPA rewrite behavior
- some basic security headers

## End-to-End Runtime Flow

Here is the normal runtime sequence:

1. User opens the frontend.
2. React starts and calls the backend for a question.
3. Backend reads from the prepared question data.
4. User selects an answer and writes an explanation.
5. Frontend sends that to `/api/submit` or Modal's `submit` endpoint.
6. Backend prediction service runs the classifier and explainer.
7. Backend returns a feedback object.
8. Frontend renders:
   - result banner
   - correct answer comparison
   - top predictions
   - teacher feedback
   - misconception card when available

## Why Dockerfile Exists

`deployment/backend/Dockerfile` is an alternative container-based way to run the backend.

The README focuses on Modal, but the Dockerfile shows that the backend can also be packaged as a normal container using:

- Python 3.11
- installed requirements
- copied `app/` and `data/`
- `uvicorn` as the server command

So the folder is designed to be portable, not tied only to one hosting platform.

## Important Notes and Caveats

### 1. Data files are required

The backend is not fully self-sufficient until `prepare_data.py` has been run.

### 2. The LoRA adapter is expected locally

Right now, real adapter loading depends on a local adapter directory inside `deployment/backend/data/adapter/`.

### 3. Frontend Modal support already exists

The frontend already contains logic for Modal URL patterns.

### 4. There are small compatibility edges

Two things to be aware of:

- the frontend has a feedback rating call, but the Modal file does not currently expose a `rate` endpoint
- the frontend has a `listQuestions` helper for Modal, but `modal_app.py` does not define a separate list endpoint

These are not necessarily fatal, especially if those paths are not used in the current UI, but they are worth remembering.

## Simple Mental Model

If you want a short mental model, think of the folder like this:

- `backend/` = the brain
- `frontend/` = the student-facing screen
- `data/` = the prepared memory the brain needs
- `modal_app.py` = the cloud wrapper for the brain
- `vercel.json` = the cloud wrapper for the screen

## Final Summary

The `deployment/` folder is a self-contained deployment package for this project.

It separates the app into:

- a React frontend for students
- a FastAPI + Qwen backend for misconception detection and feedback

The backend depends on prepared data artifacts and, for the full trained behavior, a local LoRA adapter folder. The frontend already knows how to talk to both local FastAPI and deployed Modal endpoints.
