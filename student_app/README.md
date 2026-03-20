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
