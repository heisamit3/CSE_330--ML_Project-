# Student App: How It Works

## What this app is

`student_app` is a small FastAPI application that serves a student-facing math practice interface.

It has two main parts:

- `student_app/app.py`: backend API and feedback logic
- `student_app/static/js/app.js`: frontend behavior in the browser

The app is not using a live database and it is not calling an LLM or external AI service to generate feedback. Most of the behavior is rule-based and built from examples in `DS/train.csv`.

## High-level flow

1. The browser loads `/`, which renders `templates/index.html`.
2. The frontend calls `GET /api/question` to fetch one question and its answer options.
3. The student chooses an MCQ option and writes an explanation.
4. The frontend sends that data to `POST /api/submit`.
5. The backend checks whether the choice is correct, classifies the explanation, builds feedback, and returns JSON.
6. The frontend renders the verdict, correct method, and misconception guidance.

## How the data is handled

### Source of data

At startup, the backend loads data from:

- `DS/train.csv`

The dataset rows look like this:

- `QuestionId`
- `QuestionText`
- `MC_Answer`
- `StudentExplanation`
- `Category`
- `Misconception`

So each row is not just a question. It also contains:

- one answer choice seen in the dataset
- one student explanation for that choice
- a label such as `True_Correct` or `False_Misconception`
- an optional misconception tag

### In-memory loading

When `app.py` starts, it runs `load_questions()`. That function:

- clears the global `QUESTIONS_DB`
- reads `DS/train.csv`
- groups rows by `QuestionId`
- builds one in-memory record per question

Each question record stores:

- the question text
- a list of answer options
- the inferred correct answer
- a shortlist of strong correct explanations
- an `answer_profiles` map for each option

Each answer profile contains:

- example explanations linked to that answer
- a dominant reasoning band for fallback use
- a dominant misconception for fallback use

If `DS/train.csv` is missing, the app falls back to a few built-in demo questions instead of failing.

### No persistent student storage

This app does not store student submissions in a database.

What is stored:

- questions and feedback examples in memory on the server
- theme in `localStorage` on the client
- score totals in `sessionStorage` on the client

What is not really stored yet:

- ratings from `POST /api/rate` are accepted, but not persisted anywhere
- submitted explanations are used to generate the current response only

## How the correct answer is decided

The app does not assume that every `MC_Answer` row is correct.

Instead, for each question it counts how often each answer appears with `True_*` categories. The answer with the strongest true-label support is treated as the correct answer.

That means the correct option is inferred from the dataset labels, not hardcoded from a single column.

## How the response is generated

### Important point

The response is generated locally with rules plus similarity matching against past explanations in the dataset. It is not free-form text generation from a language model.

### Step 1: Validate the submission

`POST /api/submit` receives:

- `question_id`
- `selected_answer` or `student_answer`
- `student_explanation`

The backend normalizes spaces and answer text, checks that the question exists, and verifies that the chosen answer matches one of the options for that question.

### Step 2: Decide whether the chosen option is correct

The selected answer is compared against the question's inferred correct answer after normalization.

This produces:

- correct choice
- or incorrect choice

### Step 3: Classify the student's reasoning

This is handled by `classify_reasoning()`.

The backend first finds the answer profile for the selected option. Then it compares the student's explanation with example explanations attached to that same option.

Similarity is computed from two signals:

- `SequenceMatcher` text similarity
- token overlap using a Jaccard-style score

Those are combined into one score. If example matches pass the threshold, the top matches vote for:

- reasoning band: `Correct`, `Neither`, or `Misconception`
- misconception label, when applicable

If the explanation is too short or no strong example match is found, the app falls back to heuristics. Those heuristics look at:

- minimum length
- token count
- presence of reasoning words such as `because`, `divide`, `multiply`, `fraction`, `decimal`
- whether the chosen answer's profile usually points to a misconception

### Step 4: Combine answer correctness with reasoning quality

`build_feedback()` combines:

- answer correctness: `True` or `False`
- reasoning band: `Correct`, `Neither`, or `Misconception`

This creates one category such as:

- `True_Correct`
- `True_Neither`
- `False_Correct`
- `False_Misconception`

That category is mapped to predefined feedback text, color, and headline.

### Step 5: Add misconception help and solution text

If a misconception was detected, the app pulls:

- a misconception name
- a short explanation of what went wrong
- a student-facing tip

It also adds a "correct way to solve it" explanation. That solution comes from:

1. `QUESTION_GUIDES` if the question has a custom guide
2. otherwise a strong correct explanation from the dataset
3. otherwise a simple fallback like "The correct answer is ..."

### Step 6: Return JSON to the frontend

The final JSON includes fields like:

- `is_correct`
- `selected_answer`
- `correct_answer`
- `result_headline`
- `result_message`
- `reasoning_label`
- `misconception_name`
- `misconception_tip`
- `correct_explanation`
- `confidence`

The frontend then renders those values into the result cards.

## Frontend behavior

The frontend logic in `static/js/app.js` does a few important things:

- fetches a random question from `/api/question`
- renders options as radio buttons
- requires the student to choose an option
- requires a non-empty explanation with at least 20 characters
- sends the submission as JSON
- updates the practice score in browser session storage
- displays the returned feedback panel

So the browser handles UI state, while the backend handles correctness and reasoning analysis.

## Practical summary

The app is basically a dataset-driven feedback engine wrapped in a student practice UI.

Its core pattern is:

- load labeled examples from CSV
- build an in-memory question bank
- compare a new explanation to past explanations for the chosen answer
- combine answer correctness with reasoning quality
- return a fixed, structured feedback response

That makes the app lightweight and explainable, but also means its quality depends heavily on the coverage and consistency of the dataset labels.
