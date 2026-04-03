# Paper References — MAP Kaggle Pipeline

*Covers all techniques used in `rf.ipynb`, `rf_inf_expAi.ipynb`, `3. map-qwen3-8b(training).ipynb`, and the embedded pipeline under `rf_results/`.*

---

## Competition & Task

**[C1]** Eedi. (2024). **Eedi - Mining Misconceptions in Mathematics**. Kaggle Competition.  
*The MAP@3 competition dataset: student math explanations labeled with misconception categories.*  
`https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics`

---

## 1. Foundation Models & Language Models

**[M1]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is all you need**. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.  
*The original Transformer architecture that Qwen3-8B and Qwen2.5-7B are built on.*

**[M2]** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of deep bidirectional transformers for language understanding**. *Proceedings of NAACL-HLT 2019*, 4171–4186. arXiv:1810.04805.  
*Established the sequence-classification paradigm used for misconception detection.*

**[M3]** Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., … & Zhu, T. (2023). **Qwen technical report**. arXiv:2309.16609.  
*Base architecture for Qwen3-8B used as the sequence classifier.*

**[M4]** Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., … & Zhu, T. (2024). **Qwen2 technical report**. arXiv:2407.10671.  
*Architecture and training for Qwen2.5-7B-Instruct used as the LLM scorer.*

**[M5]** Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., … & Lowe, R. (2022). **Training language models to follow instructions with human feedback**. *NeurIPS 2022*. arXiv:2203.02155.  
*Instruction-following capability behind Qwen2.5-7B-Instruct used for KB scoring and counterfactual generation.*

---

## 2. Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

**[P1]** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). **LoRA: Low-rank adaptation of large language models**. *Proceedings of ICLR 2022*. arXiv:2106.09685.  
*The LoRA adapter technique used to fine-tune Qwen3-8B for sequence classification.*

**[P2]** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient finetuning of quantized LLMs**. *NeurIPS 2023*. arXiv:2305.14314.  
*QLoRA (4-bit quantized LoRA) used to fine-tune Qwen3-8B on limited GPU memory; produces the ~200MB adapter.*

**[P3]** Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). **PEFT: State-of-the-art parameter-efficient fine-tuning methods**. Hugging Face. `https://github.com/huggingface/peft`  
*The `peft` library used to load and apply the LoRA adapter.*

**[P4]** Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). **LLM.int8(): 8-bit matrix multiplication for transformers at scale**. *NeurIPS 2022*. arXiv:2208.07339.  
*`bitsandbytes` library backend for 4-bit and 8-bit quantization in the training pipeline.*

---

## 3. Training Regularization Techniques

### 3.1 Adversarial Weight Perturbation (AWP)

**[R1]** Wu, D., Xu, S., & Dai, B. (2020). **Adversarial weight perturbation helps robust generalization**. *NeurIPS 2020*. arXiv:2004.05884.  
*AWP implementation in training — perturbs model weights to create a flatter loss surface. Explicitly cited in the training code (reported +1.0–2.0% MAP gain).*

### 3.2 R-Drop (Consistency Regularization)

**[R2]** Liang, X., Wu, L., Li, J., Wang, Y., Meng, F., Li, T., Chen, Y., Qin, T., & Liu, T.-Y. (2021). **R-Drop: Regularized dropout for neural networks**. *NeurIPS 2021*. arXiv:2106.14448.  
*Dual forward pass with symmetric KL divergence loss; RDROP_ALPHA=0.5 (reported +0.5–1.5% MAP gain).*

### 3.3 Exponential Moving Average (EMA)

**[R3]** Tarvainen, A., & Valpola, H. (2017). **Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results**. *NeurIPS 2017*. arXiv:1703.01780.  
*EMA shadow model (decay=0.999) used during and after training to smooth weight updates (reported +0.3–0.8% MAP gain).*

### 3.4 Label Smoothing

**[R4]** Müller, R., Kornblith, S., & Hinton, G. E. (2019). **When does label smoothing help?** *NeurIPS 2019*. arXiv:1906.02629.  
*Label smoothing applied to cross-entropy loss to prevent overconfident predictions.*

**[R5]** Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). **Rethinking the inception architecture for computer vision**. *CVPR 2016*.  
*Original paper introducing label smoothing as a regularizer.*

### 3.5 Optimizer & Scheduling

**[R6]** Loshchilov, I., & Hutter, F. (2019). **Decoupled weight decay regularization**. *ICLR 2019*. arXiv:1711.05101.  
*AdamW optimizer used in fine-tuning; decouples weight decay from gradient updates.*

**[R7]** Howard, J., & Ruder, S. (2018). **Universal language model fine-tuning for text classification**. *ACL 2018*. arXiv:1801.06146.  
*Popularized layerwise learning rate decay (decay_rate=0.95 per layer) used in Qwen3-8B fine-tuning.*

**[R8]** Pascanu, R., Mikolov, T., & Bengio, Y. (2013). **On the difficulty of training recurrent neural networks**. *ICML 2013*. arXiv:1211.5063.  
*Gradient clipping (max_grad_norm=1.0) to prevent exploding gradients during training.*

---

## 4. Retrieval-Augmented Generation (RAG & GraphRAG)

**[G1]** Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … & Kiela, D. (2020). **Retrieval-augmented generation for knowledge-intensive NLP tasks**. *NeurIPS 2020*. arXiv:2005.11401.  
*Core RAG paradigm behind the KB scoring stage. Explicitly cited in `retrieve_context.py`.*

**[G2]** Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., … & Larson, J. (2024). **From local to global: A graph RAG approach to query-focused summarization**. arXiv:2404.16130.  
*GraphRAG: the bounded-subgraph retrieval approach used to provide candidate-centered context from the misconception ontology graph.*

**[G3]** Jiang, J., Zhou, K., Dong, Z., Ye, K., Zhao, W. X., & Wen, J.-R. (2023). **StructGPT: A general framework for large language model to reason over structured data**. *EMNLP 2023*. arXiv:2305.09645.  
*Structured data reasoning framework that inspired the KB+graph context construction. Explicitly cited in `retrieve_context.py`.*

**[G4]** Luo, L., Li, Y.-F., Haffari, G., & Pan, S. (2024). **Reasoning on graphs: Faithful and interpretable large language model reasoning**. *ICLR 2024*. arXiv:2310.01061.  
*RoG: graph-grounded LLM reasoning used as a reference for the ontology graph scoring module. Explicitly cited in `retrieve_context.py`.*

---

## 5. Counterfactual Reasoning

**[CF1]** Wu, T., Ribeiro, M. T., Heer, J., & Weld, D. S. (2021). **Polyjuice: Generating counterfactuals for explaining, evaluating, and improving models**. *ACL 2021*. arXiv:2101.00288.  
*Counterfactual text rewriting approach: generate "what if student didn't have this misconception?" rewrites to measure causal evidence strength.*

**[CF2]** Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2019). **Counterfactual fairness in text classification through robustness**. *AIES 2019*.  
*Counterfactual perturbation methodology for measuring model sensitivity to text changes.*

**[CF3]** Wu, S., et al. (2024). **Counterfactual reasoning for large language model evaluation**. arXiv (cited in `generate_counterfactual.py`).  
*The specific counterfactual evaluation work cited in the pipeline for causal delta scoring.*

---

## 6. Knowledge Graphs & Ontologies

**[K1]** Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., de Melo, G., Gutierrez, C., … & Zimmermann, A. (2021). **Knowledge graphs**. *ACM Computing Surveys*, 54(4), 1–37. arXiv:2003.02320.  
*Foundational survey on knowledge graphs; the misconception ontology graph uses the same node/edge design patterns.*

**[K2]** Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). **Exploring network structure, dynamics, and function using NetworkX**. *Proceedings of SciPy 2008*, 11–15.  
*NetworkX library used to build and query the misconception ontology graph (GEXF export).*

**[K3]** Rittle-Johnson, B., et al. (2025). **Neuro-symbolic knowledge graphs for educational misconception modeling**. *(cited in `build_graph.py`)*  
*Referenced for the neuro-symbolic approach combining LLM reasoning with structured graph-based knowledge for educational misconception detection.*

---

## 7. Ensemble Methods & Reranking

**[E1]** Chen, T., & Guestrin, C. (2016). **XGBoost: A scalable tree boosting system**. *Proceedings of KDD 2016*, 785–794. arXiv:1603.02754.  
*XGBoost binary classifier used as the optional reranker (n_estimators=120, max_depth=4). Final blend: `0.7 * weighted_score + 0.3 * xgb_probability`.*

**[E2]** Breiman, L. (1996). **Stacking regressions**. *Machine Learning*, 24(1), 49–64.  
*Motivation for the multi-signal ensemble fusion combining 14 diverse features from classifier, KB, graph, and counterfactual stages.*

---

## 8. Text Retrieval & Feature Extraction

**[T1]** Salton, G., & Buckley, C. (1988). **Term-weighting approaches in automatic text retrieval**. *Information Processing & Management*, 24(5), 513–523.  
*TF-IDF vectorizer (n-gram range (1,2), sublinear_tf=True) used in the baseline retrieval path when pre-computed artifacts are unavailable.*

**[T2]** Manning, C. D., Raghavan, P., & Schütze, H. (2008). **Introduction to information retrieval**. Cambridge University Press.  
*Defines MAP@k (Mean Average Precision at k) — the primary competition metric. Chapter 8 covers ranked retrieval evaluation.*

---

## 9. Evaluation

**[EV1]** Järvelin, K., & Kekäläinen, J. (2002). **Cumulated gain-based evaluation of IR techniques**. *ACM Transactions on Information Systems*, 20(4), 422–446.  
*NDCG and graded relevance evaluation; the AP@3 formula in `metrics.py` derives from this framework.*

**[EV2]** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … & Duchesnay, É. (2011). **Scikit-learn: Machine learning in Python**. *Journal of Machine Learning Research*, 12, 2825–2830.  
*`StratifiedGroupKFold` used for cross-validation; `TfidfVectorizer`, `LabelEncoder`, `linear_kernel` all from scikit-learn.*

---

## 10. Educational Misconceptions (Domain Background)

**[ED1]** Booth, J. L., & Siegler, R. S. (2008). **Numerical magnitude representations influence arithmetic learning**. *Child Development*, 79(4), 1016–1031.  
*Background on numerical misconceptions in early mathematics education — relevant to the decimal/fraction misconception categories in the dataset.*

**[ED2]** Smith, J. P., diSessa, A. A., & Roschelle, J. (1994). **Misconceptions reconceived: A constructivist analysis of knowledge in transition**. *Journal of the Learning Sciences*, 3(2), 115–163.  
*Seminal work on how mathematical misconceptions form and persist — background for the KB structure (cognitive_roots, prerequisite_concepts fields).*

**[ED3]** Stacey, K., & MacGregor, M. (1997). **Ideas about symbolism that students bring to algebra**. *The Mathematics Teacher*, 90(2), 110–113.  
*Student misconceptions about algebraic notation — one of the major concept families in the dataset (`False_*` algebra categories).*

---

## 11. Explainability (XAI)

**[XAI1]** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). **"Why should I trust you?": Explaining the predictions of any classifier**. *KDD 2016*. arXiv:1602.04938.  
*LIME explainability framework; the Phase 2 Explainable AI cells in `rf_inf_expAi.ipynb` generate natural-language explanations in the same spirit.*

**[XAI2]** Lundberg, S. M., & Lee, S.-I. (2017). **A unified approach to interpreting model predictions**. *NeurIPS 2017*. arXiv:1705.07874.  
*SHAP values conceptual basis for the feature-importance weighting in the ensemble fusion (14 hand-tuned weights per feature).*

---

## 12. Additional Referenced Works (in Code Comments)

**[AD1]** Zhao, W. X., et al. (2023). **FaR: Factuality-enhanced reasoning for large language models**. *(cited in `score_candidates.py` for calibration of LLM affinity scores)*

**[AD2]** MiRAGe. *(cited in `score_candidates.py` as reference for retrieval-reranking methodology)*

**[AD3]** MalAlgoQA. *(cited in `generate_counterfactual.py` for style-preserving counterfactual generation methodology)*

---

## Quick Reference Table

| Technique Used | Paper | Where Used |
|---|---|---|
| Transformer architecture | [M1] Vaswani 2017 | Qwen3-8B, Qwen2.5-7B base |
| Sequence classification pre-training | [M2] Devlin 2019 | Qwen3-8B fine-tuning paradigm |
| Qwen3-8B model | [M3] Bai 2023 | Stage 1 classifier |
| Qwen2.5-7B-Instruct | [M4] Yang 2024 | Stage 4 LLM scorer |
| Instruction following (RLHF) | [M5] Ouyang 2022 | Qwen2.5 chat capability |
| LoRA | [P1] Hu 2022 | Qwen3-8B adapter |
| QLoRA | [P2] Dettmers 2023 | 4-bit fine-tuning |
| PEFT library | [P3] Mangrulkar 2022 | Adapter loading |
| BitsAndBytes quantization | [P4] Dettmers 2022 | 4-bit inference |
| AWP | [R1] Wu 2020 | Training regularization |
| R-Drop | [R2] Liang 2021 | Training regularization |
| EMA | [R3] Tarvainen 2017 | Model weight smoothing |
| Label smoothing | [R4] Müller 2019, [R5] Szegedy 2016 | Loss regularization |
| AdamW | [R6] Loshchilov 2019 | Optimizer |
| Layerwise LR decay | [R7] Howard 2018 | Fine-tuning schedule |
| Gradient clipping | [R8] Pascanu 2013 | Gradient stabilization |
| RAG | [G1] Lewis 2020 | KB retrieval stage |
| GraphRAG | [G2] Edge 2024 | Subgraph context retrieval |
| StructGPT | [G3] Jiang 2023 | Structured KB reasoning |
| RoG | [G4] Luo 2024 | Graph-guided LLM reasoning |
| Counterfactual NLP | [CF1] Wu 2021, [CF3] Wu 2024 | Causal delta scoring |
| Knowledge graphs | [K1] Hogan 2021 | Misconception ontology |
| NetworkX | [K2] Hagberg 2008 | Graph construction & query |
| Neuro-symbolic for education | [K3] Rittle-Johnson 2025 | Graph design |
| XGBoost | [E1] Chen 2016 | Optional reranker |
| Ensemble stacking | [E2] Breiman 1996 | Multi-signal fusion |
| TF-IDF | [T1] Salton 1988 | Baseline retrieval fallback |
| MAP@3 metric | [T2] Manning 2008 | Evaluation |
| Graded relevance (AP) | [EV1] Järvelin 2002 | MAP@3 formula |
| Scikit-learn | [EV2] Pedregosa 2011 | CV, encoding, similarity |
| Math misconceptions (domain) | [ED2] Smith 1994, [ED3] Stacey 1997 | Dataset background |
| XAI / explanations | [XAI1] Ribeiro 2016 | Phase 2 explainability |

---

*Generated for `Extra/kaggle_notebooks/` — covers `rf.ipynb`, `rf_inf_expAi.ipynb`, `3. map-qwen3-8b(training).ipynb`, and all source under `rf_results/embedded_codeds/`.*
