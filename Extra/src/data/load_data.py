from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import first_existing_path, load_dataframe


COLUMN_ALIASES = {
    "QuestionId": ["QuestionId", "question_id", "QuestionID", "id"],
    "QuestionText": ["QuestionText", "question_text", "Question"],
    "MC_Answer": ["MC_Answer", "SelectedAnswer", "Answer", "StudentAnswer"],
    "CorrectAnswer": ["CorrectAnswer", "correct_answer", "Correct_Answer"],
    "StudentExplanation": ["StudentExplanation", "student_explanation", "Explanation", "StudentResponse"],
    "Category": ["Category", "category"],
    "Misconception": ["Misconception", "misconception", "MisconceptionName"],
    "MisconceptionId": ["MisconceptionId", "misconception_id"],
    "target": ["target", "Target"],
}


def _pick_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _pick_column(list(renamed.columns), aliases)
        if found and found != canonical:
            mapping[found] = canonical
    if mapping:
        renamed = renamed.rename(columns=mapping)
    return renamed


def _infer_family_from_target(value: Any) -> str:
    text = str(value or "")
    if text.startswith("True_"):
        return "True_"
    if text.startswith("False_"):
        return "False_"
    return ""


def _ensure_standard_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = _rename_known_columns(df)
    if "QuestionId" not in frame.columns:
        frame["QuestionId"] = [f"{prefix}_q_{index}" for index in range(len(frame))]
    if "row_id" not in frame.columns:
        frame["row_id"] = [f"{prefix}_{index}" for index in range(len(frame))]
    if "StudentExplanation" not in frame.columns:
        frame["StudentExplanation"] = ""
    if "QuestionText" not in frame.columns:
        frame["QuestionText"] = ""
    if "MC_Answer" not in frame.columns:
        frame["MC_Answer"] = ""

    if "target" not in frame.columns and {"Category", "Misconception"}.issubset(frame.columns):
        frame["target"] = frame["Category"].astype(str) + ":" + frame["Misconception"].astype(str)

    if "is_correct" not in frame.columns:
        if "Category" in frame.columns:
            frame["is_correct"] = frame["Category"].astype(str).str.startswith("True_").astype(int)
        elif {"CorrectAnswer", "MC_Answer"}.issubset(frame.columns):
            frame["is_correct"] = (frame["CorrectAnswer"].astype(str) == frame["MC_Answer"].astype(str)).astype(int)
        else:
            frame["is_correct"] = pd.NA

    if "family" not in frame.columns:
        if "target" in frame.columns:
            frame["family"] = frame["target"].map(_infer_family_from_target)
        else:
            frame["family"] = frame["is_correct"].map({1: "True_", 0: "False_"}).fillna("")
    return frame


def _candidate_values(configured: Any, defaults: list[str]) -> list[str | Path]:
    values: list[str | Path] = []
    if isinstance(configured, (list, tuple)):
        values.extend(configured)
    elif configured:
        values.append(configured)
    values.extend(defaults)
    return values


def infer_correct_answer_lookup(train_df: pd.DataFrame) -> dict[str, str]:
    frame = _rename_known_columns(train_df)
    if frame.empty or "QuestionId" not in frame.columns or "MC_Answer" not in frame.columns:
        return {}

    if "CorrectAnswer" in frame.columns and frame["CorrectAnswer"].notna().any():
        correct_rows = frame.dropna(subset=["CorrectAnswer"])[["QuestionId", "CorrectAnswer"]].drop_duplicates("QuestionId")
        return dict(zip(correct_rows["QuestionId"].astype(str), correct_rows["CorrectAnswer"].astype(str)))

    if "Category" not in frame.columns:
        return {}

    true_rows = frame[frame["Category"].astype(str).str.startswith("True_")].copy()
    if true_rows.empty:
        return {}

    true_rows["freq"] = true_rows.groupby(["QuestionId", "MC_Answer"])["MC_Answer"].transform("count")
    true_rows = true_rows.sort_values(["QuestionId", "freq"], ascending=[True, False]).drop_duplicates("QuestionId")
    return dict(zip(true_rows["QuestionId"].astype(str), true_rows["MC_Answer"].astype(str)))


def apply_correctness_lookup(df: pd.DataFrame, correct_lookup: dict[str, str]) -> pd.DataFrame:
    if df.empty or not correct_lookup or "QuestionId" not in df.columns or "MC_Answer" not in df.columns:
        return df

    updated = df.copy()
    inferred_correct = updated["QuestionId"].astype(str).map(correct_lookup)
    if "CorrectAnswer" not in updated.columns:
        updated["CorrectAnswer"] = inferred_correct
    else:
        updated["CorrectAnswer"] = updated["CorrectAnswer"].fillna(inferred_correct)

    known_mask = inferred_correct.notna()
    if known_mask.any():
        updated.loc[known_mask, "is_correct"] = (
            updated.loc[known_mask, "MC_Answer"].astype(str) == inferred_correct.loc[known_mask].astype(str)
        ).astype(int)

    inferred_family = updated["is_correct"].map({1: "True_", 0: "False_"})
    if "family" not in updated.columns:
        updated["family"] = inferred_family.fillna("")
    else:
        updated["family"] = updated["family"].replace("", pd.NA).fillna(inferred_family).fillna("")
    return updated


def build_input_text(row: pd.Series | dict[str, Any]) -> str:
    if not isinstance(row, pd.Series):
        row = pd.Series(row)
    correctness = row.get("is_correct")
    if correctness == 1:
        correctness_text = "Yes"
    elif correctness == 0:
        correctness_text = "No"
    else:
        correctness_text = "Unknown"
    return "\n".join(
        [
            f"Question: {row.get('QuestionText', '')}".strip(),
            f"Answer: {row.get('MC_Answer', '')}".strip(),
            f"Correct? {correctness_text}",
            f"Student Explanation: {row.get('StudentExplanation', '')}".strip(),
        ]
    )


def _build_toy_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = [
        {
            "QuestionId": "q1",
            "QuestionText": "Which decimal is larger, 0.355 or 0.8?",
            "MC_Answer": "0.355",
            "CorrectAnswer": "0.8",
            "StudentExplanation": "0.355 is larger because 355 is bigger than 8.",
            "Category": "False_Decimal",
            "Misconception": "Uses whole-number magnitude to compare decimals",
            "target": "False_DecimalComparison:Uses whole-number magnitude to compare decimals",
        },
        {
            "QuestionId": "q1",
            "QuestionText": "Which decimal is larger, 0.355 or 0.8?",
            "MC_Answer": "0.355",
            "CorrectAnswer": "0.8",
            "StudentExplanation": "The number with more digits is bigger.",
            "Category": "False_Decimal",
            "Misconception": "Uses whole-number magnitude to compare decimals",
            "target": "False_DecimalComparison:Uses whole-number magnitude to compare decimals",
        },
        {
            "QuestionId": "q2",
            "QuestionText": "What is 1/2 + 1/3?",
            "MC_Answer": "2/5",
            "CorrectAnswer": "5/6",
            "StudentExplanation": "I add the top and bottom so it becomes 2 over 5.",
            "Category": "False_Fraction",
            "Misconception": "Adds numerators and denominators separately",
            "target": "False_FractionAddition:Adds numerators and denominators separately",
        },
        {
            "QuestionId": "q2",
            "QuestionText": "What is 1/2 + 1/3?",
            "MC_Answer": "5/6",
            "CorrectAnswer": "5/6",
            "StudentExplanation": "I made the denominators the same and then added the numerators.",
            "Category": "True_Correct",
            "Misconception": "NA",
            "target": "True_Correct:NA",
        },
        {
            "QuestionId": "q3",
            "QuestionText": "Which decimal is larger, 0.41 or 0.405?",
            "MC_Answer": "0.405",
            "CorrectAnswer": "0.41",
            "StudentExplanation": "405 is greater than 41 so 0.405 is bigger.",
            "Category": "False_Decimal",
            "Misconception": "Uses whole-number magnitude to compare decimals",
            "target": "False_DecimalComparison:Uses whole-number magnitude to compare decimals",
        },
        {
            "QuestionId": "q4",
            "QuestionText": "What is 2/7 + 1/7?",
            "MC_Answer": "3/7",
            "CorrectAnswer": "3/7",
            "StudentExplanation": "The denominators already match so I add the numerators.",
            "Category": "True_Correct",
            "Misconception": "NA",
            "target": "True_Correct:NA",
        },
    ]
    test_rows = [
        {
            "QuestionId": "t1",
            "QuestionText": "Which decimal is larger, 0.71 or 0.8?",
            "MC_Answer": "0.71",
            "CorrectAnswer": "0.8",
            "StudentExplanation": "71 is bigger than 8 so 0.71 must be larger.",
        },
        {
            "QuestionId": "t2",
            "QuestionText": "What is 1/4 + 1/2?",
            "MC_Answer": "2/6",
            "CorrectAnswer": "3/4",
            "StudentExplanation": "I added 1 plus 1 and 4 plus 2.",
        },
        {
            "QuestionId": "t3",
            "QuestionText": "What is 3/10 + 2/10?",
            "MC_Answer": "5/10",
            "CorrectAnswer": "5/10",
            "StudentExplanation": "The tenths are already the same so I combine the numerators.",
        },
    ]

    train_df = _ensure_standard_columns(pd.DataFrame(train_rows), prefix="train")
    test_df = _ensure_standard_columns(pd.DataFrame(test_rows), prefix="test")
    candidate_df = build_candidate_table(train_df, metadata_df=None)
    return train_df, test_df, candidate_df


def build_candidate_table(train_df: pd.DataFrame, metadata_df: pd.DataFrame | None = None) -> pd.DataFrame:
    candidates: pd.DataFrame
    if train_df is not None and not train_df.empty and "target" in train_df.columns:
        columns = [column for column in ["target", "Misconception", "Category", "family"] if column in train_df.columns]
        candidates = train_df[columns].drop_duplicates().copy()
        if "target" in candidates.columns:
            candidates = candidates.rename(columns={"target": "misconception_id"})
        if "Misconception" in candidates.columns:
            candidates = candidates.rename(columns={"Misconception": "misconception_name"})
        if "misconception_name" not in candidates.columns:
            candidates["misconception_name"] = candidates["misconception_id"]
        if "family" not in candidates.columns:
            if "Category" in candidates.columns:
                candidates["family"] = candidates["Category"].astype(str).str.extract(r"^(True_|False_)")[0].fillna("")
            else:
                candidates["family"] = candidates["misconception_id"].map(_infer_family_from_target)
    else:
        candidates = pd.DataFrame(columns=["misconception_id", "misconception_name", "family"])

    if metadata_df is not None and not metadata_df.empty:
        metadata = _rename_known_columns(metadata_df)
        if "misconception_id" not in metadata.columns:
            if "MisconceptionId" in metadata.columns:
                metadata = metadata.rename(columns={"MisconceptionId": "misconception_id"})
            elif "target" in metadata.columns:
                metadata = metadata.rename(columns={"target": "misconception_id"})
        if "misconception_name" not in metadata.columns and "Misconception" in metadata.columns:
            metadata = metadata.rename(columns={"Misconception": "misconception_name"})
        keep_cols = [column for column in ["misconception_id", "misconception_name", "family", "short_description"] if column in metadata.columns]
        metadata = metadata[keep_cols].drop_duplicates()
        if candidates.empty:
            candidates = metadata.copy()
        elif {"misconception_id"}.issubset(metadata.columns):
            candidates = candidates.merge(metadata, on="misconception_id", how="left", suffixes=("", "_meta"))
            if "misconception_name_meta" in candidates.columns:
                candidates["misconception_name"] = candidates["misconception_name"].fillna(candidates["misconception_name_meta"])
                candidates = candidates.drop(columns=["misconception_name_meta"])
            if "family_meta" in candidates.columns:
                candidates["family"] = candidates["family"].replace("", pd.NA).fillna(candidates["family_meta"])
                candidates = candidates.drop(columns=["family_meta"])

    if "misconception_id" not in candidates.columns:
        candidates["misconception_id"] = []
    if "misconception_name" not in candidates.columns:
        candidates["misconception_name"] = candidates["misconception_id"]
    if "family" not in candidates.columns:
        candidates["family"] = candidates["misconception_id"].map(_infer_family_from_target)
    if "short_description" not in candidates.columns:
        candidates["short_description"] = ""
    return candidates.drop_duplicates(subset=["misconception_id"]).reset_index(drop=True)


def load_competition_data(config: dict[str, Any], paths: dict[str, Path], logger) -> dict[str, pd.DataFrame]:
    input_dir = paths["input_dir"]
    train_file = config.get("paths", {}).get("train_file", "train.csv")
    test_file = config.get("paths", {}).get("test_file", "test.csv")
    misconception_file = config.get("paths", {}).get("misconception_file", "misconception_mapping.csv")

    train_path = first_existing_path(_candidate_values(train_file, ["train.csv"]), base=input_dir)
    test_path = first_existing_path(_candidate_values(test_file, ["test.csv"]), base=input_dir)
    misconception_path = first_existing_path(
        _candidate_values(misconception_file, ["misconception_mapping.csv", "misconceptions.csv"]),
        base=input_dir,
    )

    if not train_path and config.get("quick_run", False):
        logger.info("No competition CSVs found. Using built-in toy data because quick_run=true.")
        train_df, test_df, candidate_df = _build_toy_data()
        return {"train": train_df, "test": test_df, "candidates": candidate_df}

    if not train_path:
        raise FileNotFoundError(
            f"Could not find train file under {input_dir}. Set paths.train_file correctly or use quick_run mode."
        )

    train_df = _ensure_standard_columns(load_dataframe(train_path), prefix="train")
    test_df = _ensure_standard_columns(load_dataframe(test_path), prefix="test") if test_path else pd.DataFrame()
    metadata_df = _rename_known_columns(load_dataframe(misconception_path)) if misconception_path else None

    correct_lookup = infer_correct_answer_lookup(train_df)
    if not test_df.empty:
        test_df = apply_correctness_lookup(test_df, correct_lookup)

    candidate_df = build_candidate_table(train_df, metadata_df)

    max_rows = config.get("data", {}).get("max_rows")
    if max_rows:
        train_df = train_df.head(int(max_rows)).copy()
        if not test_df.empty:
            test_df = test_df.head(int(max_rows)).copy()

    logger.info(
        "Loaded train=%s rows, test=%s rows, candidates=%s",
        len(train_df),
        len(test_df),
        len(candidate_df),
    )
    return {"train": train_df, "test": test_df, "candidates": candidate_df}


def make_submission(score_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    ranked = score_df.sort_values(["row_id", "final_score"], ascending=[True, False]).copy()
    predictions = (
        ranked.groupby("row_id")["misconception_id"]
        .apply(lambda items: " ".join(str(item) for item in items.head(top_k) if pd.notna(item)))
        .reset_index()
    )
    predictions = predictions.rename(columns={"misconception_id": "prediction"})
    return predictions

