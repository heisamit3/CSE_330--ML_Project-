"""Inspired by the 45th-place-style QLoRA classifier path and the user's Final_Model notebooks.

Original idea: run notebook-driven LoRA sequence-classification inference over MAP rows and keep the
baseline submission path intact.
Simplified version here: expose that notebook logic as a small CLI that exports CSV artifacts in the
same workspace layout used by the research MVP, so the ontology/counterfactual stack can reuse them.
Why simplified: keeps Kaggle execution reproducible, restartable, and friendly to single-GPU notebooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_data import load_competition_data
from src.utils.io import first_existing_path, read_json, read_yaml, save_dataframe
from src.utils.kaggle_paths import resolve_runtime_paths
from src.utils.logging_utils import setup_logger


def _discover_search_roots(config: dict[str, Any]) -> list[Path]:
    candidates = [
        config.get("paths", {}).get("input_dir"),
        "/kaggle/input",
        "/kaggle/working",
        Path.cwd(),
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _ranked_path_sort_key(candidate: Path, preferred_fragments: list[str] | None = None) -> tuple[int, int, str]:
    preferred_fragments = preferred_fragments or []
    text = str(candidate).lower().replace('\\', '/')
    score = sum(1 for fragment in preferred_fragments if fragment and fragment.lower() in text)
    return (score, -len(text), text)


def _search_file(roots: list[Path], file_name: str, preferred_fragments: list[str] | None = None) -> Path | None:
    matches: list[Path] = []
    for root in roots:
        for candidate in root.rglob(file_name):
            if candidate.is_file():
                matches.append(candidate)
    if not matches:
        return None
    matches.sort(key=lambda candidate: _ranked_path_sort_key(candidate, preferred_fragments), reverse=True)
    return matches[0]


def _search_dir(roots: list[Path], dir_names: list[str], required_file: str | None = None, preferred_fragments: list[str] | None = None) -> Path | None:
    matches: list[Path] = []
    for root in roots:
        for dir_name in dir_names:
            for candidate in root.rglob(dir_name):
                if not candidate.is_dir():
                    continue
                if required_file and not (candidate / required_file).exists():
                    continue
                matches.append(candidate)
    if not matches:
        return None
    matches.sort(key=lambda candidate: _ranked_path_sort_key(candidate, preferred_fragments), reverse=True)
    return matches[0]


def format_classifier_input(row: pd.Series, style: str = "question_yes_no") -> str:
    if style == "statement":
        correctness_text = "This answer is correct." if row.get("is_correct") == 1 else "This answer is incorrect."
        return "\n".join(
            [
                f"Question: {row.get('QuestionText', '')}".strip(),
                f"Answer: {row.get('MC_Answer', '')}".strip(),
                correctness_text,
                f"Student Explanation: {row.get('StudentExplanation', '')}".strip(),
            ]
        )

    if row.get("is_correct") == 1:
        correctness_text = "Yes"
    elif row.get("is_correct") == 0:
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


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)


def apply_family_filter(probabilities: np.ndarray, labels: list[str], family_prefix: str) -> np.ndarray:
    if not family_prefix:
        return probabilities
    mask = np.array([str(label).startswith(family_prefix) for label in labels], dtype=bool)
    if not mask.any():
        return probabilities
    filtered = probabilities.copy()
    filtered[~mask] = 0.0
    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    return filtered


def resolve_baseline_qwen_assets(config: dict[str, Any], logger) -> dict[str, Any]:
    qcfg = config.get("baseline_qwen", {})
    search_roots = _discover_search_roots(config)

    adapter_path = first_existing_path(
        qcfg.get("adapter_candidates") or qcfg.get("adapter_path")
    )
    if adapter_path is None:
        adapter_path = _search_dir(
            search_roots,
            dir_names=["adapter"],
            required_file="adapter_config.json",
            preferred_fragments=["qwen3-8b-qlora", "trained", "map"],
        )

    label_encoder_path = first_existing_path(
        qcfg.get("label_encoder_candidates") or qcfg.get("label_encoder_path")
    )
    if label_encoder_path is None:
        label_encoder_path = _search_file(
            search_roots,
            file_name="label_encoder.joblib",
            preferred_fragments=["qwen3-8b-qlora", "trained", "map"],
        )

    run_config_path = first_existing_path(
        qcfg.get("run_config_candidates") or qcfg.get("run_config_path")
    )
    if run_config_path is None:
        run_config_path = _search_file(
            search_roots,
            file_name="run_config.json",
            preferred_fragments=["qwen3-8b-qlora", "trained", "map"],
        )
    run_config = read_json(run_config_path, default={}) if run_config_path else {}

    if adapter_path is None:
        inferred_adapter_candidates: list[Path] = []
        for metadata_path in [label_encoder_path, run_config_path]:
            if metadata_path is None:
                continue
            metadata_parent = Path(metadata_path).parent
            inferred_adapter_candidates.append(metadata_parent / "adapter")
            inferred_adapter_candidates.append(metadata_parent)
            checkpoints_dir = metadata_parent / "checkpoints"
            if checkpoints_dir.exists():
                inferred_adapter_candidates.extend(sorted(checkpoints_dir.glob("checkpoint-*"), reverse=True))

        for candidate in inferred_adapter_candidates:
            if candidate.is_dir() and (candidate / "adapter_config.json").exists():
                adapter_path = candidate
                break

    base_model_path = first_existing_path(
        qcfg.get("base_model_candidates") or qcfg.get("base_model_path")
    )
    if base_model_path is None:
        candidate_names = []
        for raw in qcfg.get("base_model_candidates") or []:
            try:
                candidate_names.append(Path(raw).name)
            except Exception:
                continue
        if run_config.get("base_model_name"):
            candidate_names.append(Path(str(run_config["base_model_name"])).name)
        candidate_names.append("Qwen3-8B")
        candidate_names = [name for index, name in enumerate(candidate_names) if name and name not in candidate_names[:index]]
        base_model_path = _search_dir(
            search_roots,
            dir_names=candidate_names,
            required_file="config.json",
            preferred_fragments=["qwen3-8b", "loaded_models", "base_models"],
        )
    base_model_name = qcfg.get("base_model_name") or run_config.get("base_model_name")

    peft_base_name = None
    if adapter_path is not None:
        try:
            from peft import PeftConfig

            peft_base_name = PeftConfig.from_pretrained(str(adapter_path), local_files_only=bool(qcfg.get("local_files_only", True))).base_model_name_or_path
        except Exception:
            peft_base_name = None

    base_model_reference = str(base_model_path) if base_model_path is not None else (base_model_name or peft_base_name or "")
    max_length = int(qcfg.get("max_length") or run_config.get("max_length") or 256)

    resolved = {
        "adapter_path": str(adapter_path) if adapter_path is not None else "",
        "label_encoder_path": str(label_encoder_path) if label_encoder_path is not None else "",
        "run_config_path": str(run_config_path) if run_config_path is not None else "",
        "run_config": run_config,
        "base_model_reference": base_model_reference,
        "max_length": max_length,
    }
    logger.info("Resolved baseline_qwen assets: %s", json.dumps({k: v for k, v in resolved.items() if k != 'run_config'}, ensure_ascii=False))
    return resolved


def prepare_split_frame(frame: pd.DataFrame, style: str, limit: int | None = None) -> pd.DataFrame:
    prepared = frame.copy()
    if limit:
        prepared = prepared.head(limit).copy()
    prepared["text"] = prepared.apply(lambda row: format_classifier_input(row, style=style), axis=1)
    return prepared


def load_label_list(train_df: pd.DataFrame, label_encoder_path: str, logger) -> list[str]:
    if label_encoder_path:
        try:
            import joblib

            label_encoder = joblib.load(label_encoder_path)
            labels = [str(value) for value in label_encoder.classes_]
            logger.info("Loaded %s labels from %s", len(labels), label_encoder_path)
            return labels
        except Exception as exc:
            logger.warning("Could not load label encoder from %s: %s", label_encoder_path, exc)

    if "target" not in train_df.columns:
        raise ValueError("Train dataframe does not contain target labels and no label encoder was found.")

    labels = sorted(train_df["target"].astype(str).dropna().unique().tolist())
    logger.info("Reconstructed %s labels directly from train.csv", len(labels))
    return labels


def load_sequence_classifier(assets: dict[str, Any], labels: list[str], config: dict[str, Any], logger):
    qcfg = config.get("baseline_qwen", {})
    if not assets.get("base_model_reference"):
        raise FileNotFoundError("Could not resolve the Qwen base model. Set baseline_qwen.base_model_candidates.")
    if not assets.get("adapter_path"):
        raise FileNotFoundError("Could not resolve the LoRA adapter. Set baseline_qwen.adapter_candidates.")

    num_labels = len(labels)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_files_only = bool(qcfg.get("local_files_only", True))
    device = qcfg.get("device", "cuda")
    dtype_name = qcfg.get("dtype", "float16")
    dtype_value = getattr(torch, dtype_name, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        assets["base_model_reference"],
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}

    model_kwargs: dict[str, Any] = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "ignore_mismatched_sizes": True,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": dtype_value,
    }
    if device != "cpu":
        model_kwargs["device_map"] = "auto"
    if qcfg.get("use_4bit", False):
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype_value,
            )
        except Exception as exc:
            logger.warning("4-bit load requested but BitsAndBytesConfig was unavailable: %s", exc)

    logger.info("Loading base sequence classifier from %s", assets["base_model_reference"])
    base_model = AutoModelForSequenceClassification.from_pretrained(
        assets["base_model_reference"],
        **model_kwargs,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    base_model.config.problem_type = "single_label_classification"

    logger.info("Loading LoRA adapter from %s", assets["adapter_path"])
    model = PeftModel.from_pretrained(
        base_model,
        assets["adapter_path"],
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def run_model_inference(frame: pd.DataFrame, labels: list[str], model, tokenizer, config: dict[str, Any], max_length: int, logger) -> np.ndarray:
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    qcfg = config.get("baseline_qwen", {})
    batch_size = int(qcfg.get("batch_size", 8))
    logger.info("Running sequence-classification inference on %s rows with batch_size=%s", len(frame), batch_size)

    dataset = Dataset.from_dict({"text": frame["text"].tolist()})

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    dataset = dataset.with_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(model_device) for key, value in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    logger.info("Collected logits with shape=%s for %s labels", logits.shape, len(labels))
    return softmax_numpy(logits)


def export_prediction_artifacts(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
    config: dict[str, Any],
    paths: dict[str, Path],
    split_name: str,
    logger,
) -> dict[str, Path]:
    qcfg = config.get("baseline_qwen", {})
    output_prefix = qcfg.get("output_prefix", "qwen_adapter")
    top_k_export = int(qcfg.get("top_k_export", config.get("data", {}).get("top_k_candidates", 12)))
    family_filter = bool(qcfg.get("family_filter", True))

    wide_records: list[dict[str, Any]] = []
    long_records: list[dict[str, Any]] = []

    for row_index, (_, row) in enumerate(frame.iterrows()):
        row_prob = probabilities[row_index]
        if family_filter:
            row_prob = apply_family_filter(row_prob, labels, str(row.get("family", "") or ""))
        top_indices = np.argsort(row_prob)[::-1][:top_k_export]
        prediction_parts: list[str] = []
        wide_row: dict[str, Any] = {"row_id": row["row_id"], "family": row.get("family", "")}

        for rank, class_index in enumerate(top_indices, start=1):
            misconception_id = str(labels[class_index])
            score = float(row_prob[class_index])
            wide_row[f"label_{rank}"] = misconception_id
            wide_row[f"prob_{rank}"] = score
            if rank <= 3:
                prediction_parts.append(misconception_id)
            long_records.append(
                {
                    "row_id": row["row_id"],
                    "misconception_id": misconception_id,
                    "baseline_score": score,
                }
            )

        wide_row["prediction"] = " ".join(prediction_parts)
        wide_records.append(wide_row)

    wide_df = pd.DataFrame(wide_records)
    long_df = pd.DataFrame(long_records).sort_values(["row_id", "baseline_score"], ascending=[True, False])
    submission_df = wide_df[["row_id", "prediction"]].copy()

    wide_path = paths["baseline_artifacts_dir"] / f"{output_prefix}_{split_name}_topk.csv"
    long_path = paths["baseline_artifacts_dir"] / f"{output_prefix}_{split_name}_scores_long.csv"
    submission_path = paths["output_dir"] / f"{output_prefix}_{split_name}_submission.csv"
    metadata_path = paths["output_dir"] / f"{output_prefix}_{split_name}_metadata.json"

    save_dataframe(wide_df, wide_path, index=False)
    save_dataframe(long_df, long_path, index=False)
    save_dataframe(submission_df, submission_path, index=False)
    metadata = {
        "split": split_name,
        "rows": int(len(frame)),
        "labels": int(len(labels)),
        "family_filter": family_filter,
        "top_k_export": top_k_export,
        "artifact_files": {
            "wide_topk": str(wide_path),
            "long_scores": str(long_path),
            "submission": str(submission_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved baseline artifacts for %s to %s", split_name, long_path)
    return {
        "wide_topk": wide_path,
        "long_scores": long_path,
        "submission": submission_path,
        "metadata": metadata_path,
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_h100_qwen_full.yaml")
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    paths = resolve_runtime_paths(config)
    logger = setup_logger("qwen_adapter_infer", paths["logs_dir"])
    logger.info("Starting qwen_adapter_infer with config=%s", args.config)

    data_bundle = load_competition_data(config, paths, logger)
    train_df = data_bundle["train"]
    test_df = data_bundle["test"]

    style = config.get("baseline_qwen", {}).get("input_format_style", "question_yes_no")
    requested_splits = [args.split] if args.split != "both" else ["train", "test"]

    prepared_frames: dict[str, pd.DataFrame] = {}
    for split_name in requested_splits:
        source_frame = train_df if split_name == "train" else test_df
        if source_frame.empty:
            logger.warning("Skipping %s because the dataframe is empty.", split_name)
            continue
        prepared = prepare_split_frame(source_frame, style=style, limit=args.limit)
        preview_path = paths["output_dir"] / f"qwen_adapter_{split_name}_prepared_preview.csv"
        preview_columns = [column for column in ["row_id", "QuestionId", "MC_Answer", "is_correct", "family", "text"] if column in prepared.columns]
        save_dataframe(prepared[preview_columns].head(32), preview_path, index=False)
        logger.info("Prepared %s rows for %s. Preview saved to %s", len(prepared), split_name, preview_path)
        prepared_frames[split_name] = prepared

    if args.prepare_only:
        logger.info("Prepare-only mode finished without model loading.")
        return

    assets = resolve_baseline_qwen_assets(config, logger)
    labels = load_label_list(train_df, assets.get("label_encoder_path", ""), logger)
    model, tokenizer = load_sequence_classifier(assets, labels=labels, config=config, logger=logger)

    for split_name, prepared in prepared_frames.items():
        probabilities = run_model_inference(
            prepared,
            labels,
            model,
            tokenizer,
            config=config,
            max_length=assets["max_length"],
            logger=logger,
        )
        export_prediction_artifacts(prepared, probabilities, labels, config, paths, split_name, logger)


if __name__ == "__main__":
    run()

