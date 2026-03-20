import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def lines(text: str):
    return [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()]


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
}


def build_training_cells():
    cells = [
        md_cell(
            """
            # MAP Multi-LLM Training Notebook

            This notebook is a practical, writeup-inspired training/export workflow for the Kaggle MAP competition.

            It follows the core ideas described in the 45th-place silver writeup:

            - shared preprocessing with `is_correct`
            - classifier-style fine-tuning on structured question/answer/explanation text
            - support for LoRA, QLoRA, and full fine-tuning presets
            - export of full test-set probability matrices for later ensembling
            - family-aware MAP@3 validation (`True_` vs `False_`)

            How to use it:

            1. Pick one `ACTIVE_MODEL_KEY`.
            2. Run the notebook top to bottom.
            3. Change `ACTIVE_MODEL_KEY` and run again for the next model.
            4. Create one Kaggle dataset from the exported `artifacts/` folder.
            5. Use `map_multi_llm_submission.ipynb` to ensemble all saved runs.

            Notes:

            - The silver writeup used a 6-model ensemble and strong hardware. This notebook gives you the same workflow shape, but some presets will still require more VRAM than a free Kaggle GPU.
            - The writeup lists a Qwen2 8B run; this notebook uses the public `Qwen/Qwen2-7B-Instruct` checkpoint as the closest widely available Qwen2 preset.
            """
        ),
        code_cell(
            """
            !pip install -q -U "transformers>=4.55.0" peft accelerate bitsandbytes datasets scikit-learn scipy
            """
        ),
        code_cell(
            """
            import gc
            import inspect
            import json
            import random
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from datasets import Dataset
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            from scipy.special import softmax
            from sklearn.model_selection import train_test_split
            from transformers import (
                AutoModel,
                AutoTokenizer,
                BitsAndBytesConfig,
                DataCollatorWithPadding,
                Trainer,
                TrainingArguments,
            )
            from transformers.modeling_outputs import SequenceClassifierOutput

            print("CUDA available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
                print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
            """
        ),
        code_cell(
            """
            # =========================
            # CONFIG
            # =========================
            DEBUG = False
            SEED = 42
            RUN_NAME_SUFFIX = ""
            VAL_SIZE = 0.10
            RUN_FINAL_FULL_TRAIN = True
            SAVE_PROBABILITIES_AS_FLOAT16 = True

            TRAIN_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
            TEST_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"

            OUTPUT_ROOT = Path("/kaggle/working/map_multi_llm_runs")
            ARTIFACT_ROOT = OUTPUT_ROOT / "artifacts"
            ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

            MODEL_PRESETS = {
                "qwen3_4b_lora": {
                    "run_slug": "qwen3_4b_lora",
                    "model_id": "Qwen/Qwen3-4B",
                    "method": "lora",
                    "max_length": 256,
                    "epochs": 3,
                    "learning_rate": 2e-5,
                    "train_batch_size": 1,
                    "eval_batch_size": 4,
                    "gradient_accumulation_steps": 16,
                    "weight_decay": 0.30,
                    "lora_r": 512,
                    "lora_alpha": 512,
                    "lora_dropout": 0.05,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "ensemble_weight": 1.0,
                },
                "hunyuan_7b_lora": {
                    "run_slug": "hunyuan_7b_lora",
                    "model_id": "tencent/Hunyuan-7B-Instruct",
                    "method": "lora",
                    "max_length": 256,
                    "epochs": 3,
                    "learning_rate": 2e-4,
                    "train_batch_size": 1,
                    "eval_batch_size": 4,
                    "gradient_accumulation_steps": 16,
                    "weight_decay": 0.01,
                    "lora_r": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.05,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "ensemble_weight": 1.0,
                },
                "qwen2_7b_qlora": {
                    "run_slug": "qwen2_7b_qlora",
                    "model_id": "Qwen/Qwen2-7B-Instruct",
                    "method": "qlora",
                    "max_length": 256,
                    "epochs": 2,
                    "learning_rate": 2e-4,
                    "train_batch_size": 1,
                    "eval_batch_size": 4,
                    "gradient_accumulation_steps": 16,
                    "weight_decay": 0.01,
                    "lora_r": 64,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "ensemble_weight": 1.0,
                },
                "qwen3_14b_qlora": {
                    "run_slug": "qwen3_14b_qlora",
                    "model_id": "Qwen/Qwen3-14B",
                    "method": "qlora",
                    "max_length": 192,
                    "epochs": 1,
                    "learning_rate": 2e-4,
                    "train_batch_size": 2,
                    "eval_batch_size": 8,
                    "gradient_accumulation_steps": 8,
                    "weight_decay": 0.01,
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "run_final_full_train": False,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "ensemble_weight": 1.0,
                },
                "qwen3_8b_full": {
                    "run_slug": "qwen3_8b_full",
                    "model_id": "Qwen/Qwen3-8B",
                    "method": "full",
                    "max_length": 256,
                    "epochs": 3,
                    "learning_rate": 2e-5,
                    "train_batch_size": 1,
                    "eval_batch_size": 2,
                    "gradient_accumulation_steps": 32,
                    "weight_decay": 0.01,
                    "ensemble_weight": 1.0,
                },
                "deepseek_math_7b_full": {
                    "run_slug": "deepseek_math_7b_full",
                    "model_id": "deepseek-ai/deepseek-math-7b-instruct",
                    "method": "full",
                    "max_length": 256,
                    "epochs": 3,
                    "learning_rate": 2e-5,
                    "train_batch_size": 1,
                    "eval_batch_size": 2,
                    "gradient_accumulation_steps": 32,
                    "weight_decay": 0.01,
                    "ensemble_weight": 1.0,
                },
            }

            ACTIVE_MODEL_KEY = "qwen3_4b_lora"
            PRESET = MODEL_PRESETS[ACTIVE_MODEL_KEY].copy()
            RUN_FINAL_FULL_TRAIN = PRESET.get("run_final_full_train", RUN_FINAL_FULL_TRAIN)

            if DEBUG:
                PRESET["epochs"] = 1
                PRESET["gradient_accumulation_steps"] = min(2, PRESET["gradient_accumulation_steps"])

            RUN_SLUG = f"{PRESET['run_slug']}_seed{SEED}"
            if RUN_NAME_SUFFIX:
                RUN_SLUG = f"{RUN_SLUG}_{RUN_NAME_SUFFIX}"

            RUN_DIR = ARTIFACT_ROOT / RUN_SLUG
            RUN_DIR.mkdir(parents=True, exist_ok=True)

            print("Available presets:", ", ".join(MODEL_PRESETS.keys()))
            print(json.dumps(PRESET, indent=2))
            print("Run slug:", RUN_SLUG)
            print("RUN_FINAL_FULL_TRAIN:", RUN_FINAL_FULL_TRAIN)
            print("Artifacts will be written to:", RUN_DIR)
            """
        ),
    ]
    cells.extend(
        [
            code_cell(
                """
                # =========================
                # HELPERS
                # =========================
                def seed_everything(seed: int = 42):
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)


                def resolve_hidden_size(config):
                    for attr in ["hidden_size", "d_model", "n_embd", "model_dim"]:
                        value = getattr(config, attr, None)
                        if value is not None:
                            return int(value)
                    raise ValueError("Could not infer hidden size from model config.")


                def build_correct_answer_map(train_df: pd.DataFrame) -> dict:
                    idx = train_df["Category"].astype(str).str.split("_").str[0] == "True"
                    correct = train_df.loc[idx].copy()
                    correct["count_per_pair"] = correct.groupby(["QuestionId", "MC_Answer"])["MC_Answer"].transform("count")
                    correct = correct.sort_values("count_per_pair", ascending=False).drop_duplicates(["QuestionId"])
                    return {
                        int(qid): str(ans)
                        for qid, ans in zip(correct["QuestionId"], correct["MC_Answer"])
                    }


                def add_correctness_features(df: pd.DataFrame, correct_by_q: dict) -> pd.DataFrame:
                    df = df.copy()
                    df["QuestionText"] = df["QuestionText"].fillna("")
                    df["StudentExplanation"] = df["StudentExplanation"].fillna("")
                    df["MC_Answer"] = df["MC_Answer"].fillna("")
                    df["correct_answer"] = df["QuestionId"].map(
                        lambda qid: correct_by_q.get(int(qid)) if pd.notna(qid) else None
                    )
                    df["is_correct"] = np.where(
                        df["correct_answer"].notna(),
                        (df["MC_Answer"] == df["correct_answer"]).astype(int),
                        pd.NA,
                    )
                    df["family_prefix"] = df["is_correct"].map(
                        lambda x: None if pd.isna(x) else ("True_" if int(x) == 1 else "False_")
                    )
                    return df


                def format_input(row: pd.Series) -> str:
                    if pd.isna(row["is_correct"]):
                        correct_text = "Unknown"
                    else:
                        correct_text = "Yes" if int(row["is_correct"]) == 1 else "No"
                    return (
                        f"Question: {row['QuestionText']}\\n"
                        f"Answer: {row['MC_Answer']}\\n"
                        f"Is Correct Answer: {correct_text}\\n"
                        f"Student Explanation: {row['StudentExplanation']}"
                    )


                def build_family_fillers(classes, train_df):
                    fillers = {}
                    target_counts = train_df["target"].value_counts()
                    for prefix in ["True_", "False_"]:
                        family_classes = [c for c in classes if c.startswith(prefix)]
                        family_classes = sorted(
                            family_classes,
                            key=lambda c: (-target_counts.get(c, 0), c),
                        )
                        if prefix == "True_":
                            preferred = [c for c in ["True_Correct:NA", "True_Neither:NA"] if c in family_classes]
                        else:
                            preferred = [c for c in ["False_Neither:NA"] if c in family_classes]
                        seen = set(preferred)
                        fillers[prefix] = preferred + [c for c in family_classes if c not in seen]
                    return fillers


                def map3_from_top3(pred_top3, true_labels):
                    total = 0.0
                    for pred, truth in zip(pred_top3, true_labels):
                        for rank, label in enumerate(pred[:3], start=1):
                            if label == truth:
                                total += 1.0 / rank
                                break
                    return total / max(len(true_labels), 1)


                def select_top3(score_row, classes, family_prefix=None, family_fillers=None):
                    ranked = sorted(zip(classes, score_row), key=lambda x: x[1], reverse=True)
                    if family_prefix:
                        ranked = [item for item in ranked if item[0].startswith(family_prefix)]
                    selected = [label for label, _ in ranked[:3]]
                    if family_prefix and family_fillers:
                        for filler in family_fillers.get(family_prefix, []):
                            if filler not in selected:
                                selected.append(filler)
                            if len(selected) == 3:
                                break
                    if len(selected) < 3:
                        for label, _ in sorted(zip(classes, score_row), key=lambda x: x[1], reverse=True):
                            if label not in selected:
                                selected.append(label)
                            if len(selected) == 3:
                                break
                    return selected[:3]


                def evaluate_prediction_matrix(probs, labels, family_prefixes, classes, family_fillers):
                    raw_top3 = [select_top3(row, classes) for row in probs]
                    filtered_top3 = [
                        select_top3(
                            row,
                            classes,
                            family_prefix=family_prefixes[i],
                            family_fillers=family_fillers,
                        )
                        for i, row in enumerate(probs)
                    ]
                    true_labels = [classes[int(idx)] for idx in labels]
                    return {
                        "map3_raw": map3_from_top3(raw_top3, true_labels),
                        "map3_family": map3_from_top3(filtered_top3, true_labels),
                        "raw_top3": raw_top3,
                        "filtered_top3": filtered_top3,
                    }


                def metric_map3(eval_pred):
                    logits, labels = eval_pred
                    probs = softmax(logits, axis=1)
                    top3 = np.argsort(-probs, axis=1)[:, :3]
                    score = 0.0
                    for i in range(len(labels)):
                        for rank, pred_idx in enumerate(top3[i], start=1):
                            if pred_idx == labels[i]:
                                score += 1.0 / rank
                                break
                    return {"map3": score / max(len(labels), 1)}


                def build_dataset(df, tokenizer, max_length, include_labels=True):
                    columns = ["text"] + (["label"] if include_labels else [])
                    ds = Dataset.from_pandas(df[columns].copy(), preserve_index=False)
                    ds = ds.map(
                        lambda batch: tokenizer(
                            batch["text"],
                            truncation=True,
                            max_length=max_length,
                        ),
                        batched=True,
                        remove_columns=["text"],
                    )
                    return ds


                def estimate_warmup_steps(num_rows, per_device_batch_size, gradient_accumulation_steps, epochs, ratio=0.1):
                    effective_batch = max(
                        int(per_device_batch_size) * max(int(gradient_accumulation_steps), 1),
                        1,
                    )
                    steps_per_epoch = max((int(num_rows) + effective_batch - 1) // effective_batch, 1)
                    total_steps = max(int(steps_per_epoch * float(epochs)), 1)
                    return max(int(total_steps * float(ratio)), 1)


                def make_training_arguments(**kwargs):
                    params = inspect.signature(TrainingArguments.__init__).parameters
                    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
                        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
                    if "evaluation_strategy" in params and "eval_strategy" in kwargs:
                        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

                    filtered = {key: value for key, value in kwargs.items() if key in params}
                    return TrainingArguments(**filtered)


                def make_trainer(**kwargs):
                    params = inspect.signature(Trainer.__init__).parameters
                    if "processing_class" in params and "tokenizer" in kwargs:
                        kwargs["processing_class"] = kwargs.pop("tokenizer")
                    if "tokenizer" in params and "processing_class" in kwargs:
                        kwargs["tokenizer"] = kwargs.pop("processing_class")

                    filtered = {key: value for key, value in kwargs.items() if key in params}
                    return Trainer(**filtered)


                seed_everything(SEED)
                """
            ),
            code_cell(
                """
                # =========================
                # LOAD + PREPROCESS
                # =========================
                train = pd.read_csv(TRAIN_PATH)
                test = pd.read_csv(TEST_PATH)

                train["Misconception"] = train["Misconception"].fillna("NA")
                train["target"] = train["Category"] + ":" + train["Misconception"]

                classes = sorted(train["target"].unique().tolist())
                class_to_id = {label: idx for idx, label in enumerate(classes)}
                train["label"] = train["target"].map(class_to_id).astype(int)

                correct_by_q = build_correct_answer_map(train)
                train = add_correctness_features(train, correct_by_q)
                test = add_correctness_features(test, correct_by_q)

                train["text"] = train.apply(format_input, axis=1)
                test["text"] = test.apply(format_input, axis=1)

                if DEBUG:
                    train = train.sample(n=min(4000, len(train)), random_state=SEED).reset_index(drop=True)

                family_fillers = build_family_fillers(classes, train)

                print("Train shape:", train.shape)
                print("Test shape:", test.shape)
                print("Classes:", len(classes))
                print("Sample prompt:")
                print(train.loc[0, "text"][:800])
                """
            ),
            code_cell(
                """
                # =========================
                # LEAK-SAFE VALIDATION SPLIT
                # =========================
                counts = train["label"].value_counts()
                singleton_labels = set(counts[counts == 1].index.tolist())

                singleton_df = train[train["label"].isin(singleton_labels)].copy()
                multi_df = train[~train["label"].isin(singleton_labels)].copy()

                train_main, val_df = train_test_split(
                    multi_df,
                    test_size=VAL_SIZE,
                    random_state=SEED,
                    stratify=multi_df["label"],
                )

                train_df = (
                    pd.concat([train_main, singleton_df], ignore_index=True)
                    .sample(frac=1.0, random_state=SEED)
                    .reset_index(drop=True)
                )
                val_df = val_df.reset_index(drop=True)

                print("Singleton labels kept in train only:", len(singleton_labels))
                print("Train rows:", len(train_df))
                print("Val rows:", len(val_df))
                """
            ),
            code_cell(
                """
                # =========================
                # TOKENIZER + DATASETS
                # =========================
                tokenizer = AutoTokenizer.from_pretrained(
                    PRESET["model_id"],
                    trust_remote_code=True,
                    use_fast=False,
                )
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({"pad_token": "<pad>"})
                tokenizer.padding_side = "right"

                train_ds = build_dataset(train_df, tokenizer, PRESET["max_length"], include_labels=True)
                val_ds = build_dataset(val_df, tokenizer, PRESET["max_length"], include_labels=True)
                test_ds = build_dataset(test, tokenizer, PRESET["max_length"], include_labels=False)

                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                print(train_ds)
                print(val_ds)
                print(test_ds)
                """
            ),
            code_cell(
                """
                # =========================
                # MODEL FACTORY
                # =========================
                class LastTokenClassifier(nn.Module):
                    supports_gradient_checkpointing = True

                    def __init__(self, backbone, hidden_size, num_labels, dropout=0.0):
                        super().__init__()
                        self.backbone = backbone
                        self.dropout = nn.Dropout(dropout)
                        self.classifier = nn.Linear(hidden_size, num_labels)
                        self.num_labels = num_labels
                        self.config = backbone.config
                        self.config.num_labels = num_labels
                        self.config.problem_type = "single_label_classification"

                    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                        if hasattr(self.backbone, "gradient_checkpointing_enable"):
                            kwargs = gradient_checkpointing_kwargs or {}
                            try:
                                self.backbone.gradient_checkpointing_enable(
                                    gradient_checkpointing_kwargs=kwargs
                                )
                            except TypeError:
                                self.backbone.gradient_checkpointing_enable()

                    def gradient_checkpointing_disable(self):
                        if hasattr(self.backbone, "gradient_checkpointing_disable"):
                            self.backbone.gradient_checkpointing_disable()

                    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                        outputs = self.backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            **kwargs,
                        )
                        hidden = outputs.last_hidden_state
                        if attention_mask is None:
                            pooled = hidden[:, -1]
                        else:
                            last_token_idx = attention_mask.sum(dim=1) - 1
                            pooled = hidden[
                                torch.arange(hidden.size(0), device=hidden.device),
                                last_token_idx,
                            ]
                        logits = self.classifier(self.dropout(pooled))
                        loss = None
                        if labels is not None:
                            loss = F.cross_entropy(logits.float(), labels)
                        return SequenceClassifierOutput(
                            loss=loss,
                            logits=logits,
                            hidden_states=outputs.hidden_states,
                            attentions=outputs.attentions,
                        )


                def count_trainable_parameters(model):
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in model.parameters())
                    return trainable, total


                def build_model(num_labels: int):
                    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    compute_dtype = (
                        torch.bfloat16 if use_bf16 else (
                            torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                    )

                    load_in_4bit = PRESET["method"] == "qlora"
                    quant_config = None
                    if load_in_4bit and torch.cuda.is_available():
                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=compute_dtype,
                        )

                    backbone = AutoModel.from_pretrained(
                        PRESET["model_id"],
                        trust_remote_code=True,
                        torch_dtype=compute_dtype,
                        quantization_config=quant_config,
                        device_map="auto" if quant_config is not None else None,
                    )
                    backbone.resize_token_embeddings(len(tokenizer))
                    backbone.config.pad_token_id = tokenizer.pad_token_id
                    backbone.config.use_cache = False

                    if hasattr(backbone, "gradient_checkpointing_enable"):
                        backbone.gradient_checkpointing_enable()

                    if quant_config is not None:
                        backbone = prepare_model_for_kbit_training(backbone)

                    if PRESET["method"] in {"lora", "qlora"}:
                        lora_config = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            r=PRESET["lora_r"],
                            lora_alpha=PRESET["lora_alpha"],
                            lora_dropout=PRESET["lora_dropout"],
                            bias="none",
                            target_modules=PRESET["target_modules"],
                        )
                        backbone = get_peft_model(backbone, lora_config)
                        backbone.print_trainable_parameters()

                    model = LastTokenClassifier(
                        backbone=backbone,
                        hidden_size=resolve_hidden_size(backbone.config),
                        num_labels=num_labels,
                        dropout=0.0,
                    )

                    trainable, total = count_trainable_parameters(model)
                    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
                    return model, use_bf16


                model, use_bf16 = build_model(len(classes))
                """
            ),
            code_cell(
                """
                # =========================
                # TRAIN VALIDATION MODEL
                # =========================
                train_warmup_steps = estimate_warmup_steps(
                    num_rows=len(train_df),
                    per_device_batch_size=PRESET["train_batch_size"],
                    gradient_accumulation_steps=PRESET["gradient_accumulation_steps"],
                    epochs=PRESET["epochs"],
                    ratio=0.1,
                )

                train_args = make_training_arguments(
                    output_dir=str(RUN_DIR / "trainer_validation"),
                    overwrite_output_dir=True,
                    num_train_epochs=PRESET["epochs"],
                    learning_rate=PRESET["learning_rate"],
                    per_device_train_batch_size=PRESET["train_batch_size"],
                    per_device_eval_batch_size=PRESET["eval_batch_size"],
                    gradient_accumulation_steps=PRESET["gradient_accumulation_steps"],
                    weight_decay=PRESET["weight_decay"],
                    lr_scheduler_type="cosine",
                    warmup_steps=train_warmup_steps,
                    eval_strategy="epoch",
                    save_strategy="no",
                    logging_steps=25 if DEBUG else 100,
                    bf16=use_bf16,
                    fp16=torch.cuda.is_available() and not use_bf16,
                    gradient_checkpointing=True,
                    group_by_length=True,
                    dataloader_num_workers=2,
                    remove_unused_columns=False,
                    report_to="none",
                    optim="paged_adamw_8bit" if PRESET["method"] in {"lora", "qlora"} else "adamw_torch",
                )

                trainer = make_trainer(
                    model=model,
                    args=train_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=metric_map3,
                )

                trainer.train()
                """
            ),
            code_cell(
                """
                # =========================
                # VALIDATION SUMMARY
                # =========================
                val_output = trainer.predict(val_ds)
                val_probs = softmax(val_output.predictions, axis=1)
                val_summary = evaluate_prediction_matrix(
                    probs=val_probs,
                    labels=val_df["label"].to_numpy(),
                    family_prefixes=val_df["family_prefix"].tolist(),
                    classes=classes,
                    family_fillers=family_fillers,
                )

                print("Validation MAP@3 (raw):", round(val_summary["map3_raw"], 5))
                print("Validation MAP@3 (family filtered):", round(val_summary["map3_family"], 5))

                oof = val_df[["row_id", "target", "family_prefix"]].copy()
                oof["pred_raw_top3"] = [" ".join(x) for x in val_summary["raw_top3"]]
                oof["pred_family_top3"] = [" ".join(x) for x in val_summary["filtered_top3"]]
                oof.to_csv(RUN_DIR / "validation_predictions.csv", index=False)

                validation_summary = {
                    "run_slug": RUN_SLUG,
                    "active_model_key": ACTIVE_MODEL_KEY,
                    "seed": SEED,
                    "model_id": PRESET["model_id"],
                    "method": PRESET["method"],
                    "validation_map3_raw": float(val_summary["map3_raw"]),
                    "validation_map3_family": float(val_summary["map3_family"]),
                }
                (RUN_DIR / "validation_summary.json").write_text(
                    json.dumps(validation_summary, indent=2),
                    encoding="utf-8",
                )

                oof.head()
                """
            ),
            code_cell(
                """
                # =========================
                # FINAL FULL-DATA TRAIN
                # =========================
                export_trainer = trainer
                export_note = "best validation model"

                if RUN_FINAL_FULL_TRAIN:
                    full_train_df = train.copy().reset_index(drop=True)
                    full_train_ds = build_dataset(full_train_df, tokenizer, PRESET["max_length"], include_labels=True)

                    del trainer, model, val_output
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    final_model, final_use_bf16 = build_model(len(classes))
                    full_warmup_steps = estimate_warmup_steps(
                        num_rows=len(full_train_df),
                        per_device_batch_size=PRESET["train_batch_size"],
                        gradient_accumulation_steps=PRESET["gradient_accumulation_steps"],
                        epochs=PRESET["epochs"],
                        ratio=0.1,
                    )
                    final_args = make_training_arguments(
                        output_dir=str(RUN_DIR / "trainer_full"),
                        overwrite_output_dir=True,
                        num_train_epochs=PRESET["epochs"],
                        learning_rate=PRESET["learning_rate"],
                        per_device_train_batch_size=PRESET["train_batch_size"],
                        per_device_eval_batch_size=PRESET["eval_batch_size"],
                        gradient_accumulation_steps=PRESET["gradient_accumulation_steps"],
                        weight_decay=PRESET["weight_decay"],
                        lr_scheduler_type="cosine",
                        warmup_steps=full_warmup_steps,
                        eval_strategy="no",
                        save_strategy="no",
                        logging_steps=25 if DEBUG else 100,
                        bf16=final_use_bf16,
                        fp16=torch.cuda.is_available() and not final_use_bf16,
                        gradient_checkpointing=True,
                        group_by_length=True,
                        dataloader_num_workers=2,
                        remove_unused_columns=False,
                        report_to="none",
                        optim="paged_adamw_8bit" if PRESET["method"] in {"lora", "qlora"} else "adamw_torch",
                    )

                    export_trainer = make_trainer(
                        model=final_model,
                        args=final_args,
                        train_dataset=full_train_ds,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                    )
                    export_trainer.train()
                    export_note = "final full-data model"

                print("Export source:", export_note)
                """
            ),
            code_cell(
                """
                # =========================
                # TEST INFERENCE + ARTIFACT EXPORT
                # =========================
                test_output = export_trainer.predict(test_ds)
                test_probs = softmax(test_output.predictions, axis=1)

                preview_top3 = [
                    select_top3(
                        score_row=test_probs[i],
                        classes=classes,
                        family_prefix=test.loc[i, "family_prefix"],
                        family_fillers=family_fillers,
                    )
                    for i in range(len(test))
                ]

                probability_dtype = np.float16 if SAVE_PROBABILITIES_AS_FLOAT16 else np.float32
                np.savez_compressed(
                    RUN_DIR / "test_probabilities.npz",
                    row_id=test["row_id"].to_numpy(),
                    probabilities=test_probs.astype(probability_dtype),
                )

                preview = pd.DataFrame({
                    "row_id": test["row_id"],
                    "Category:Misconception": [" ".join(x) for x in preview_top3],
                })
                preview.to_csv(RUN_DIR / "submission_preview.csv", index=False)

                metadata = {
                    "run_slug": RUN_SLUG,
                    "active_model_key": ACTIVE_MODEL_KEY,
                    "seed": SEED,
                    "model_id": PRESET["model_id"],
                    "method": PRESET["method"],
                    "max_length": PRESET["max_length"],
                    "ensemble_weight": PRESET.get("ensemble_weight", 1.0),
                    "classes": classes,
                    "family_fillers": family_fillers,
                    "correct_answer_map": {str(k): v for k, v in correct_by_q.items()},
                    "validation_map3_raw": float(validation_summary["validation_map3_raw"]),
                    "validation_map3_family": float(validation_summary["validation_map3_family"]),
                    "probability_file": "test_probabilities.npz",
                    "preview_file": "submission_preview.csv",
                }
                (RUN_DIR / "metadata.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )

                print("Saved run artifacts to:", RUN_DIR)
                print("Files:")
                for path in sorted(RUN_DIR.iterdir()):
                    print(" -", path.name)

                print()
                print("Currently available run folders:")
                for path in sorted(ARTIFACT_ROOT.iterdir()):
                    if path.is_dir():
                        print(" -", path.name)

                preview.head()
                """
            ),
        ]
    )
    return cells


def build_submission_cells():
    cells = [
        md_cell(
            """
            # MAP Multi-LLM Submission Notebook

            This notebook builds the final competition submission from the probability artifacts produced by `map_multi_llm_training.ipynb`.

            Expected setup:

            - attach the competition dataset
            - attach one Kaggle dataset made from the exported `artifacts/` directory
            - each run should live in its own folder under that artifact root

            The ensemble follows the silver-writeup pattern:

            - average weighted class probabilities
            - add an agreement bonus based on top-K overlap
            - add a max-confidence bonus
            - family-filter the final ranking (`True_` vs `False_`)
            """
        ),
        code_cell(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd

            print("Notebook ready.")
            """
        ),
        code_cell(
            """
            # =========================
            # EDIT THESE PATHS
            # =========================
            COMPETITION_DATA_DIR = Path("/kaggle/input/map-charting-student-math-misunderstandings")
            ARTIFACT_ROOT = Path("/kaggle/input/map-multi-llm-artifacts")
            OUTPUT_PATH = Path("/kaggle/working/submission.csv")

            # Set to a list like ["qwen3_4b_lora", "qwen2_7b_qlora"] if you want a subset.
            SELECT_RUN_SLUGS = None
            TOP_N_BY_VALIDATION = None
            MIN_VALIDATION_MAP3 = None
            USE_VALIDATION_SCORE_WEIGHTS = True
            VALIDATION_WEIGHT_POWER = 4.0

            BASE_SCORE_WEIGHT = 0.34
            AGREEMENT_WEIGHT = 0.33
            CONFIDENCE_WEIGHT = 0.33
            VOTE_TOP_K = 10
            """
        ),
        code_cell(
            """
            # =========================
            # HELPERS
            # =========================
            def discover_run_dirs(artifact_root: Path, selected_slugs=None):
                run_dirs = []
                for path in sorted(artifact_root.iterdir()):
                    if path.is_dir() and (path / "metadata.json").exists() and (path / "test_probabilities.npz").exists():
                        run_dirs.append(path)
                if selected_slugs:
                    selected = set(selected_slugs)
                    run_dirs = [path for path in run_dirs if path.name in selected]
                if not run_dirs:
                    raise FileNotFoundError("No run folders were found under ARTIFACT_ROOT.")
                return run_dirs


            def build_family_prefix(test_df: pd.DataFrame, correct_by_q: dict) -> pd.Series:
                correct_answer = test_df["QuestionId"].map(
                    lambda qid: correct_by_q.get(int(qid)) if pd.notna(qid) else None
                )
                is_correct = np.where(
                    correct_answer.notna(),
                    (test_df["MC_Answer"].fillna("") == correct_answer).astype(int),
                    pd.NA,
                )
                family_prefix = pd.Series(is_correct).map(
                    lambda x: None if pd.isna(x) else ("True_" if int(x) == 1 else "False_")
                )
                return family_prefix


            def select_top3(score_row, classes, family_prefix=None, family_fillers=None):
                ranked = sorted(zip(classes, score_row), key=lambda x: x[1], reverse=True)
                if family_prefix:
                    ranked = [item for item in ranked if item[0].startswith(family_prefix)]
                selected = [label for label, _ in ranked[:3]]
                if family_prefix and family_fillers:
                    for filler in family_fillers.get(family_prefix, []):
                        if filler not in selected:
                            selected.append(filler)
                        if len(selected) == 3:
                            break
                if len(selected) < 3:
                    for label, _ in sorted(zip(classes, score_row), key=lambda x: x[1], reverse=True):
                        if label not in selected:
                            selected.append(label)
                        if len(selected) == 3:
                            break
                return selected[:3]


            def reorder_probs_to_master(row_ids, probs, master_row_ids):
                if np.array_equal(row_ids, master_row_ids):
                    return probs
                lookup = {int(row_id): idx for idx, row_id in enumerate(row_ids.tolist())}
                order = [lookup[int(row_id)] for row_id in master_row_ids.tolist()]
                return probs[order]


            def align_classes_to_master(probs, source_classes, master_classes):
                if source_classes == master_classes:
                    return probs
                source_index = {label: idx for idx, label in enumerate(source_classes)}
                take = [source_index[label] for label in master_classes]
                return probs[:, take]
            """
        ),
        code_cell(
            """
            # =========================
            # LOAD TEST + RUN METADATA
            # =========================
            test = pd.read_csv(COMPETITION_DATA_DIR / "test.csv")
            run_dirs = discover_run_dirs(ARTIFACT_ROOT, SELECT_RUN_SLUGS)

            print("Using run folders:")
            for path in run_dirs:
                print(" -", path.name)

            run_metadata = []
            for path in run_dirs:
                metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
                metadata["run_dir"] = path
                run_metadata.append(metadata)

            master_metadata = run_metadata[0]
            master_classes = master_metadata["classes"]
            family_fillers = master_metadata["family_fillers"]
            correct_by_q = {int(k): v for k, v in master_metadata["correct_answer_map"].items()}
            master_row_ids = test["row_id"].to_numpy()
            family_prefixes = build_family_prefix(test, correct_by_q).tolist()

            summary = pd.DataFrame(
                {
                    "run_slug": [meta["run_slug"] for meta in run_metadata],
                    "model_id": [meta["model_id"] for meta in run_metadata],
                    "method": [meta["method"] for meta in run_metadata],
                    "seed": [meta.get("seed") for meta in run_metadata],
                    "val_map3_family": [meta.get("validation_map3_family") for meta in run_metadata],
                    "ensemble_weight": [meta.get("ensemble_weight", 1.0) for meta in run_metadata],
                }
            ).sort_values("val_map3_family", ascending=False)

            if MIN_VALIDATION_MAP3 is not None:
                keep = set(summary.loc[summary["val_map3_family"] >= MIN_VALIDATION_MAP3, "run_slug"])
                run_metadata = [meta for meta in run_metadata if meta["run_slug"] in keep]
                summary = summary[summary["run_slug"].isin(keep)].copy()

            if TOP_N_BY_VALIDATION is not None:
                keep = set(summary.head(TOP_N_BY_VALIDATION)["run_slug"])
                run_metadata = [meta for meta in run_metadata if meta["run_slug"] in keep]
                summary = summary[summary["run_slug"].isin(keep)].copy()

            if not run_metadata:
                raise ValueError("No runs remain after validation-score filtering.")

            summary = summary.sort_values("val_map3_family", ascending=False).reset_index(drop=True)
            summary
            """
        ),
        code_cell(
            """
            # =========================
            # LOAD PROBABILITY MATRICES
            # =========================
            loaded_runs = []

            for meta in run_metadata:
                bundle = np.load(meta["run_dir"] / meta["probability_file"])
                probs = bundle["probabilities"].astype(np.float32)
                row_ids = bundle["row_id"]

                probs = reorder_probs_to_master(row_ids, probs, master_row_ids)
                probs = align_classes_to_master(probs, meta["classes"], master_classes)

                loaded_runs.append(
                    {
                        "run_slug": meta["run_slug"],
                        "base_weight": float(meta.get("ensemble_weight", 1.0)),
                        "val_map3_family": meta.get("validation_map3_family"),
                        "probs": probs,
                    }
                )

            print("Loaded", len(loaded_runs), "probability matrices.")
            print("Matrix shape:", loaded_runs[0]["probs"].shape)
            """
        ),
        code_cell(
            """
            # =========================
            # ENSEMBLE
            # =========================
            n_rows = len(master_row_ids)
            n_classes = len(master_classes)

            prob_sum = np.zeros((n_rows, n_classes), dtype=np.float32)
            agreement = np.zeros((n_rows, n_classes), dtype=np.float32)
            max_confidence = np.zeros((n_rows, n_classes), dtype=np.float32)

            total_weight = 0.0
            row_index = np.arange(n_rows)[:, None]

            for run in loaded_runs:
                probs = run["probs"]
                weight = run["base_weight"]
                if USE_VALIDATION_SCORE_WEIGHTS and run["val_map3_family"] is not None:
                    weight *= float(run["val_map3_family"]) ** VALIDATION_WEIGHT_POWER
                run["effective_weight"] = weight
                total_weight += weight

                prob_sum += probs * weight
                max_confidence = np.maximum(max_confidence, probs)

                k = min(VOTE_TOP_K, probs.shape[1])
                topk_idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
                agreement[row_index, topk_idx] += weight

            weight_table = pd.DataFrame(
                {
                    "run_slug": [run["run_slug"] for run in loaded_runs],
                    "val_map3_family": [run["val_map3_family"] for run in loaded_runs],
                    "effective_weight": [run["effective_weight"] for run in loaded_runs],
                }
            ).sort_values("effective_weight", ascending=False)
            weight_table

            prob_component = prob_sum / max(total_weight, 1e-8)
            agreement_component = agreement / max(total_weight, 1e-8)

            final_scores = (
                prob_component * BASE_SCORE_WEIGHT
                + agreement_component * AGREEMENT_WEIGHT
                + max_confidence * CONFIDENCE_WEIGHT
            )

            top3_predictions = [
                select_top3(
                    score_row=final_scores[i],
                    classes=master_classes,
                    family_prefix=family_prefixes[i],
                    family_fillers=family_fillers,
                )
                for i in range(n_rows)
            ]

            print("Example predictions:")
            for preds in top3_predictions[:3]:
                print(preds)
            """
        ),
        code_cell(
            """
            # =========================
            # BUILD SUBMISSION
            # =========================
            submission = pd.DataFrame(
                {
                    "Category:Misconception": [" ".join(preds) for preds in top3_predictions]
                }
            )
            submission.to_csv(OUTPUT_PATH, index=False)

            print("Saved:", OUTPUT_PATH)
            submission.head()
            """
        ),
    ]
    return cells


def write_notebook(path: Path, cells):
    notebook = {
        "cells": cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    training_path = ROOT / "map_multi_llm_training.ipynb"
    submission_path = ROOT / "map_multi_llm_submission.ipynb"

    write_notebook(training_path, build_training_cells())
    write_notebook(submission_path, build_submission_cells())

    print("Generated notebooks:")
    print(f" - {training_path.name}")
    print(f" - {submission_path.name}")


if __name__ == "__main__":
    main()
