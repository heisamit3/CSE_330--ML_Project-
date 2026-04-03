# Paper Usage Map — What Is Used & Where

*Maps every referenced paper to the exact file and role it plays in the pipeline.*

---

## [M1] Vaswani et al. (2017) — Attention Is All You Need

**What:** The Transformer architecture (self-attention, positional encoding, encoder-decoder blocks).

**Where used:**
| File | Usage |
|---|---|
| `rf_results/embedded_codeds/src/baseline/qwen_adapter_infer.py` | Qwen3-8B is a decoder-only Transformer loaded as `AutoModelForSequenceClassification`; the attention mechanism processes the student explanation input |
| `3. map-qwen3-8b(training).ipynb` | Full Transformer model is fine-tuned via QLoRA; the forward pass through all attention layers produces logits over misconception classes |

---

## [M2] Devlin et al. (2019) — BERT

**What:** Pre-training + fine-tuning paradigm for sequence classification; `[CLS]` token aggregation for classification.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | The classification head on top of Qwen3-8B follows the BERT-style sequence classification pattern: linear layer on the final hidden state → softmax over all misconception labels |
| `src/baseline/qwen_adapter_infer.py` | Input format mirrors BERT fine-tuning: structured text prompt fed as a single sequence, output is class probabilities |

---

## [M3] Bai et al. (2023) — Qwen Technical Report

**What:** Qwen3-8B model weights, tokenizer, and architecture.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | Model ID `Qwen/Qwen3-8B` loaded as base; fine-tuned with QLoRA for misconception classification |
| `src/baseline/qwen_adapter_infer.py` | Loads the same Qwen3-8B base + LoRA adapter at inference time; runs classification to produce top-12 candidates per row |
| `rf.ipynb` | Stage 1 of the full pipeline: Qwen3-8B adapter inference |

---

## [M4] Yang et al. (2024) — Qwen2 Technical Report

**What:** Qwen2.5-7B-Instruct model weights and instruction-following capability.

**Where used:**
| File | Usage |
|---|---|
| `src/graphrag/score_candidates.py` | `Qwen/Qwen2.5-7B-Instruct` loaded as CausalLM; called to score each candidate misconception against the student explanation using structured KB context |
| `src/counterfactual/generate_counterfactual.py` | Same model generates counterfactual rewrites: "rewrite this explanation as if the student didn't have misconception X" |
| `rf_inf_expAi.ipynb` | Phase 2: loaded again in CausalLM mode to generate natural-language explanations of predictions |
| `configs/kaggle_h100_qwen_full.yaml` | Configured as `llm.model: Qwen3-8B-Instruct` (or Qwen2.5-7B-Instruct depending on config); `device: cuda`, `precision: bfloat16` |

---

## [M5] Ouyang et al. (2022) — InstructGPT / RLHF

**What:** RLHF-based instruction tuning that makes Qwen2.5-7B-Instruct follow JSON output instructions reliably.

**Where used:**
| File | Usage |
|---|---|
| `src/graphrag/prompt_templates.py` | Prompts are structured as instruction-following requests (JSON output format); the model's RLHF training is what makes it obey the strict `{"affinity_score": ..., "confidence": ..., "reasoning_short": ...}` format |
| `src/graphrag/score_candidates.py` | Relies on instruction following to return parseable JSON; `json_repair: true` fallback needed when the model deviates |
| `src/counterfactual/generate_counterfactual.py` | Instruction: "rewrite the explanation preserving tone and length" — RLHF tuning is what makes the model follow length/style constraints |

---

## [P1] Hu et al. (2022) — LoRA

**What:** Low-Rank Adaptation — inserts trainable rank-r matrices into attention layers; only adapts ~0.1% of parameters.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)` applied to Qwen3-8B attention layers; only the adapter is trained |
| `src/baseline/qwen_adapter_infer.py` | `PeftModel.from_pretrained(base_model, adapter_path)` merges the LoRA adapter for inference; adapter is ~200MB vs full model's ~16GB |

---

## [P2] Dettmers et al. (2023) — QLoRA

**What:** Combines 4-bit NF4 quantization of the frozen base model with LoRA fine-tuning; enables fine-tuning large models on single GPUs.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16)` applied before LoRA; entire training runs in 4-bit quantized mode |
| `src/baseline/qwen_adapter_infer.py` | Optional 4-bit loading at inference time to fit within Kaggle's 16GB VRAM limit |

---

## [P3] Mangrulkar et al. (2022) — PEFT Library

**What:** HuggingFace `peft` library implementing LoRA, QLoRA, and adapter utilities.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `from peft import LoraConfig, get_peft_model, TaskType` — wraps Qwen3-8B for training |
| `src/baseline/qwen_adapter_infer.py` | `from peft import PeftModel` — loads saved adapter from `adapter/adapter_model.safetensors` |
| `rf_results/embedded_codeds/requirements.txt` | `peft>=0.12.0` listed as dependency |

---

## [P4] Dettmers et al. (2022) — LLM.int8() / BitsAndBytes

**What:** 8-bit and 4-bit quantized matrix multiplication; `bitsandbytes` library.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `bitsandbytes` backend for `BitsAndBytesConfig`; enables NF4 4-bit quantization during QLoRA training |
| `rf_results/embedded_codeds/requirements.txt` | `bitsandbytes>=0.43.0` listed as dependency |

---

## [R1] Wu et al. (2020) — AWP (Adversarial Weight Perturbation)

**What:** Perturbs model weights by a small epsilon during training to flatten the loss surface and improve generalization.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | AWP class implemented inline: after epoch 1 warmup, each training step perturbs weights by `lr=0.1, eps=1e-4`, computes perturbed loss, then restores weights. Reported +1.0–2.0% MAP gain. |

**Implementation detail:**
```
awp_start_epoch = 1
awp_lr = 0.1
awp_epsilon = 1e-4
# After normal backward pass, perturb weights → second forward/backward → restore weights
```

---

## [R2] Liang et al. (2021) — R-Drop

**What:** Runs two forward passes through the model with dropout active; minimizes KL divergence between the two output distributions to force consistency.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `RDROP_ALPHA = 0.5`; each training step: two forward passes → `loss = (loss1 + loss2)/2 + 0.5 * symmetric_KL(p1, p2)`. Reported +0.5–1.5% MAP gain. |

---

## [R3] Tarvainen & Valpola (2017) — Mean Teacher / EMA

**What:** Maintains an exponential moving average of model weights; the EMA model is more stable than the instantaneous checkpoint.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `EMA(model, decay=0.999)` shadow copy updated after each step; at evaluation and final save, EMA weights are applied. Reported +0.3–0.8% MAP gain. |

---

## [R4] Müller et al. (2019) + [R5] Szegedy et al. (2016) — Label Smoothing

**What:** Replaces hard one-hot targets with soft distributions `(1 - ε) * one_hot + ε/K` to prevent overconfident predictions.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | Applied to cross-entropy loss during fine-tuning on misconception labels; prevents the model from being overconfident on the ~2000-class classification head |

---

## [R6] Loshchilov & Hutter (2019) — AdamW

**What:** Adam optimizer with decoupled weight decay (weight decay not applied to the gradient-adaptive scaling term).

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `AdamW` from HuggingFace `transformers`; standard optimizer for LLM fine-tuning with linear warmup + decay schedule |

---

## [R7] Howard & Ruder (2018) — ULMFiT / Layerwise LR Decay

**What:** Different learning rates per layer — higher layers (closer to output) use higher LR; lower layers use lower LR to preserve pre-trained representations.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | Layerwise LR decay with `decay_rate=0.95` per layer; deeper Transformer layers use exponentially smaller LRs |

---

## [R8] Pascanu et al. (2013) — Gradient Clipping

**What:** Clips gradient norm to a maximum value to prevent exploding gradients.

**Where used:**
| File | Usage |
|---|---|
| `3. map-qwen3-8b(training).ipynb` | `max_grad_norm=1.0` applied before every optimizer step |

---

## [G1] Lewis et al. (2020) — RAG

**What:** Retrieval-Augmented Generation: retrieve relevant documents from a knowledge store, prepend to LLM context, generate.

**Where used:**
| File | Usage |
|---|---|
| `src/graphrag/retrieve_context.py` | Core RAG pattern: for each candidate misconception, retrieve its KB entry + graph neighbors → build a context block → feed to LLM scorer. Explicitly cited in docstring. |
| `src/graphrag/score_candidates.py` | The KB-guided scoring mode is RAG: retrieved context is prepended to the scoring prompt |
| `src/ontology/build_kb.py` | Builds the knowledge store (`misconception_kb.json`) that RAG retrieves from |

---

## [G2] Edge et al. (2024) — GraphRAG

**What:** Extends RAG with a knowledge graph; retrieval follows graph edges to surface related entities, not just nearest-neighbor embeddings.

**Where used:**
| File | Usage |
|---|---|
| `src/graphrag/retrieve_context.py` | `top_k_context_neighbors=3` — retrieves misconception KB entry + up to 3 graph-neighbor misconceptions via NetworkX edge traversal; bounded subgraph (max 6 nodes) to avoid noise |
| `src/ontology/graph_utils.py` | Builds the NetworkX graph with 7 edge types (`causes`, `exacerbates`, `shares_cognitive_root_with`, `shares_prerequisite`, `related_trigger`, `possible_confusion_with`, `alias_overlap`) |
| `src/graphrag/score_candidates.py` | `subgraph_density` and `graph_neighbor_count` are output features from the GraphRAG context retrieval |

---

## [G3] Jiang et al. (2023) — StructGPT

**What:** Framework for LLMs to reason over structured data (tables, KBs, graphs) by converting structure into natural-language context.

**Where used:**
| File | Usage |
|---|---|
| `src/graphrag/retrieve_context.py` | KB entries are structured objects (JSON with `trigger_patterns`, `concept_tags`, `cognitive_roots`, etc.); StructGPT's approach of flattening structured fields into prompt text is followed. Explicitly cited in docstring. |
| `src/graphrag/prompt_templates.py` | Prompt construction converts the structured KB entry + graph edges into natural-language context blocks for the LLM |

---

## [G4] Luo et al. (2024) — RoG (Reasoning on Graphs)

**What:** LLM reasoning grounded by graph structure — the graph provides faithful, interpretable paths for the model to follow.

**Where used:**
| File | Usage |
|---|---|
| `src/graphrag/retrieve_context.py` | Graph neighbor relationships (edge types and directions) are included in the LLM context; the LLM is expected to reason along these paths when scoring candidates. Explicitly cited in docstring. |
| `src/graphrag/score_candidates.py` | `ontology_support_score` feature: measures how strongly graph paths support the candidate — RoG's faithfulness concept applied to misconception scoring |

---

## [CF1] Wu et al. (2021) — Polyjuice (Counterfactuals)

**What:** Generate minimal counterfactual text edits that change model predictions; measure how sensitive the model is to specific spans.

**Where used:**
| File | Usage |
|---|---|
| `src/counterfactual/generate_counterfactual.py` | Generates "if the student didn't have this misconception" rewrites: removes trigger-pattern-heavy phrases while preserving wording/tone/length; computes `style_preservation_score` as token overlap between original and rewrite |
| `src/counterfactual/score_counterfactual.py` | Scores the rewritten explanation; computes `causal_delta = original_affinity - counterfactual_affinity`; large delta = strong evidence the misconception caused the explanation |

---

## [CF3] Wu et al. (2024) — Counterfactual LLM Evaluation

**What:** Counterfactual reasoning to evaluate causal evidence for LLM predictions.

**Where used:**
| File | Usage |
|---|---|
| `src/counterfactual/generate_counterfactual.py` | Cited for the `causal_verdict` logic: `confirmed` (causal_delta > threshold), `weak_support`, or `spurious_or_unstable` |
| `src/ensemble/fuse_scores.py` | `causal_delta` (weight 0.14) and `relative_delta` (weight 0.08) are among the highest-weighted features in the ensemble fusion |

---

## [K1] Hogan et al. (2021) — Knowledge Graphs Survey

**What:** Defines knowledge graph components: nodes, typed edges, ontologies, and reasoning over graphs.

**Where used:**
| File | Usage |
|---|---|
| `src/ontology/build_graph.py` | Misconception ontology graph follows KG design: misconception nodes, typed directed edges, GEXF export |
| `src/ontology/graph_utils.py` | Node attributes match KG schema: `misconception_id`, `family`, `concept_tags`, `trigger_patterns`; edges have `relation` type attribute |
| `src/ontology/build_kb.py` | `graph_links` field in each KB entry stores the typed edges to other misconceptions |

---

## [K2] Hagberg et al. (2008) — NetworkX

**What:** Python library for building, analyzing, and querying graphs.

**Where used:**
| File | Usage |
|---|---|
| `src/ontology/graph_utils.py` | `nx.DiGraph()` constructs the directed misconception graph; `nx.write_gexf()` exports to `graph.gexf` |
| `src/graphrag/retrieve_context.py` | `nx.ego_graph()` or neighbor traversal used to retrieve bounded subgraphs for each candidate |
| `src/eval/ablation_report.py` | Graph statistics (connected components, density) computed via NetworkX for reporting |

---

## [K3] Rittle-Johnson et al. (2025) — Educational Knowledge Graphs

**What:** Neuro-symbolic knowledge graphs for modeling student misconceptions in education.

**Where used:**
| File | Usage |
|---|---|
| `src/ontology/build_graph.py` | Cited as domain reference for the edge relation types, especially `shares_cognitive_root_with` and `causes` — which mirror cognitive science models of how math misconceptions relate |
| `src/ontology/build_kb.py` | `cognitive_roots` and `remediation_hint` KB fields are motivated by educational psychology models of misconception formation and correction |

---

## [E1] Chen & Guestrin (2016) — XGBoost

**What:** Gradient-boosted decision trees; fast, accurate binary classifier.

**Where used:**
| File | Usage |
|---|---|
| `src/ensemble/fuse_scores.py` | Optional reranker: `XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic')`; trained on 14 features to predict whether a candidate is the correct misconception |
| `src/eval/run_cv.py` | XGBoost reranker trained within each CV fold to avoid leakage; blend: `final_score = 0.7 * weighted_score + 0.3 * xgb_probability` |

---

## [E2] Breiman (1996) — Ensemble Stacking

**What:** Combining multiple model outputs using a meta-learner rather than simple averaging.

**Where used:**
| File | Usage |
|---|---|
| `src/ensemble/fuse_scores.py` | 14-feature weighted fusion is the stacking layer: baseline classifier, LLM KB scorer, graph features, and counterfactual features are all first-level signals; XGBoost is the optional meta-learner |

---

## [T1] Salton & Buckley (1988) — TF-IDF

**What:** Term frequency–inverse document frequency weighting for text retrieval; cosine similarity via linear kernel.

**Where used:**
| File | Usage |
|---|---|
| `src/baseline/pipeline.py` | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` + `linear_kernel` for cosine similarity; used as the fallback candidate retrieval path when Qwen3-8B adapter artifacts are not available |

---

## [T2] Manning et al. (2008) — Introduction to Information Retrieval / MAP

**What:** Defines Mean Average Precision (MAP) and ranked retrieval evaluation.

**Where used:**
| File | Usage |
|---|---|
| `src/eval/metrics.py` | `map_at_k(predictions, ground_truth, k=3)` implemented exactly as defined: AP@k = (1/k) * Σ(hits_at_i / i); credit 1.0 / 0.5 / 0.33 for ranks 1–3 |
| `src/eval/run_cv.py` | MAP@3 is the primary CV optimization target; ablation report shows MAP@3 per pipeline stage |

---

## [EV1] Järvelin & Kekäläinen (2002) — Graded Relevance / NDCG

**What:** Cumulative gain-based evaluation; AP@k formula derives from this work.

**Where used:**
| File | Usage |
|---|---|
| `src/eval/metrics.py` | The position-discounted credit (rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33) follows the graded relevance discount concept |

---

## [EV2] Pedregosa et al. (2011) — Scikit-learn

**What:** Machine learning library for Python: preprocessing, cross-validation, metrics.

**Where used:**
| File | Usage |
|---|---|
| `src/baseline/pipeline.py` | `TfidfVectorizer`, `linear_kernel` (cosine similarity) |
| `3. map-qwen3-8b(training).ipynb` | `StratifiedGroupKFold` for cross-validation; `LabelEncoder` mapping misconception labels to numeric indices; `train_test_split` |
| `src/baseline/qwen_adapter_infer.py` | Loads `label_encoder.joblib` (sklearn `LabelEncoder`) to convert model output indices back to label strings |

---

## [ED2] Smith et al. (1994) + [ED3] Stacey & MacGregor (1997) — Math Misconceptions

**What:** Domain theory of how mathematical misconceptions form, persist, and relate to prerequisite knowledge gaps.

**Where used:**
| File | Usage |
|---|---|
| `src/ontology/build_kb.py` | KB schema fields `cognitive_roots`, `prerequisite_concepts`, `remediation_hint` are directly motivated by constructivist misconception theory |
| `src/ontology/build_graph.py` | Edge types `causes` and `shares_prerequisite` model the prerequisite-dependency structure described in these papers |

---

## [XAI1] Ribeiro et al. (2016) — LIME

**What:** Explains model predictions by generating local perturbations and observing output changes.

**Where used:**
| File | Usage |
|---|---|
| `rf_inf_expAi.ipynb` | Phase 2 Explainable AI: Qwen3-8B is loaded in CausalLM mode and prompted to generate natural-language explanations for its predictions — same intent as LIME (understand why a specific prediction was made), implemented as LLM self-explanation |

---

## [XAI2] Lundberg & Lee (2017) — SHAP

**What:** Unified feature importance via Shapley values.

**Where used:**
| File | Usage |
|---|---|
| `src/ensemble/fuse_scores.py` | The 14 hand-tuned feature weights (`baseline_score: 0.18`, `kb_score: 0.15`, `causal_delta: 0.14`, ...) serve the same role as SHAP values — quantifying each signal's contribution to the final ranking |
| `src/eval/ablation_report.py` | Ablation study measures MAP@3 drop when each feature is removed — empirical version of SHAP importance |

---

## Summary: Pipeline Stage → Papers

| Pipeline Stage | File | Papers |
|---|---|---|
| **Stage 1: Classification** | `qwen_adapter_infer.py` | [M1][M2][M3][P1][P2][P3] |
| **Training** | `3. map-qwen3-8b(training).ipynb` | [M1][M2][M3][P1][P2][P3][P4][R1][R2][R3][R4][R5][R6][R7][R8] |
| **Stage 2: KB Construction** | `build_kb.py` | [K1][ED2][ED3] |
| **Stage 3: Graph Building** | `build_graph.py`, `graph_utils.py` | [G2][K1][K2][K3][ED2] |
| **Stage 4a: GraphRAG Retrieval** | `retrieve_context.py` | [G1][G2][G3][G4][K2] |
| **Stage 4b: LLM Scoring** | `score_candidates.py`, `prompt_templates.py` | [M4][M5][G1][G3][G4] |
| **Stage 4c: Counterfactual** | `generate_counterfactual.py`, `score_counterfactual.py` | [M4][CF1][CF3] |
| **Stage 5: Ensemble Fusion** | `fuse_scores.py` | [E1][E2][CF3] |
| **Baseline Fallback** | `pipeline.py` | [T1][EV2] |
| **Evaluation** | `metrics.py`, `run_cv.py` | [T2][EV1][EV2] |
| **Explainability** | `rf_inf_expAi.ipynb` | [M4][XAI1][XAI2] |
