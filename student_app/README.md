# MAP Student Practice App

This app is the student-facing practice UI for your project.

Students now:

- read one math question
- choose from four MCQ options
- explain their reasoning
- receive a verdict on the choice
- see the correct method
- see a misconception note when the explanation suggests one

## What changed

- The backend now builds each question from all answer choices found in `DS/train.csv` instead of treating `MC_Answer` as always correct.
- The correct option is inferred from the dataset labels.
- The feedback engine combines answer correctness with explanation matching so the verdict fits both the selected option and the written reasoning.
- The frontend flow is now explicitly `MCQ -> explanation -> verdict`.

## Run locally

```bash
cd student_app
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8001`.

## Easiest deployment without Docker

Use Vercel with your GitHub repo. This project now includes a Vercel entrypoint in `api/index.py`, a root `requirements.txt`, and a `vercel.json` file so you can deploy without Docker.

### Steps

1. Push this repository to GitHub.
2. Sign in to Vercel and import that GitHub repo as a new project.
3. Keep the default project root as the repository root.
4. Deploy the project.
5. Open the deployed URL.

### What Vercel will use

- Python entrypoint: `api/index.py`
- Dependencies: root `requirements.txt`
- Routing: root `vercel.json`

Notes:

- The deployed app reads the dataset from `DS/train.csv`.
- If `DS/train.csv` is missing, the app still works by falling back to built-in example questions.
- Static UI files remain available at `/static/...`.
- The Vercel config excludes the large training notebooks from the serverless bundle.

## Deploy with Docker

Build from the repository root so the container can include both `student_app/` and `DS/`:

```bash
docker build -f student_app/Dockerfile -t map-student-app .
docker run -p 8001:8001 map-student-app
```

Then open `http://localhost:8001`.

Notes:

- The container reads the dataset from `DS/train.csv`.
- The container respects the `PORT` environment variable, so it can run on Docker-friendly platforms that inject their own port.
- If you deploy to a cloud host, point it at `student_app/Dockerfile` and use the repository root as the build context.

## Main endpoints

- `GET /api/question`: returns one question plus its MCQ options
- `POST /api/submit`: accepts `question_id`, `selected_answer`, and `student_explanation`
- `POST /api/rate`: stores a simple helpfulness rating placeholder

## Notes

- If `DS/train.csv` is missing, the app falls back to a few built-in example questions.
- In this workspace, runtime verification was limited because `fastapi` is not installed in the shell environment yet, but the Python files compile successfully.
