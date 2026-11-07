# CS-781 Capstone: Legal Language Models and Legal NLP Tasks

Individual capstone project (CS-781, IIT Kanpur) by Siddhant Singh Rathoria (Roll No. 261110062). We pretrain four small Indian-legal language models from scratch (Phase 1) and adapt them, using parameter-efficient and full fine-tuning, to a sequence of legal NLP tasks (Phases 2-7). All training uses only the official competition data; no external corpora or pretrained weights.

## Phase 1 models (all trained from scratch, shared 32k byte-level BPE tokenizer)
| Name | Architecture | Notes |
|---|---|---|
| `llama` | Decoder-only, RMSNorm, RoPE, SwiGLU, bias-free | Llama-style |
| `gpt2` | Decoder-only, LayerNorm, learned positions, GELU MLP | GPT-2-style |
| `eff` | Decoder-only with local (sliding-window) attention, periodic global layers, grouped-query attention | Efficiency-oriented |
| `encdec` | Encoder-decoder (5+5 layers, d=768, RoPE, cross-attention) | Its encoder is reused as an encoder-only backbone (`enc`) for classification/retrieval |

## Repository layout
| Folder | Task | Main script |
|---|---|---|
| `phase1_pretraining/` | Legal LM pretraining + generation | `prep.py`, `lm.py`, `eff.py`, `encdec.py`, `gen.py` |
| `phase2_ner/` | Legal named-entity recognition | `ner_train.py` |
| `phase3_rhetorical_roles/` | Rhetorical role labelling | `p3_train.py`, `ens.py` |
| `phase4_cjpe/` | Court judgment prediction (binary, long documents) | `p4_train.py` |
| `phase5_lsi/` | Legal statute identification (multi-label) | `p5_train.py` |
| `phase6_summarization/` | Judgment summarization (encoder-decoder) | `p6_train.py` |
| `phase7_pcr/` | Prior case retrieval | `p7_train.py`, `fuse.py`, `lex.py` |

Each folder is self-contained (copies of the model files it needs) and has its own README with the method, hyperparameters, commands and results. Scripts were run on Kaggle (2x T4); paths such as `/kaggle/working` and `/kaggle/input` are hard-coded at the top of each script. The Hugging Face dataset token is read from the `HF_TOKEN` environment variable.

## Adaptation techniques implemented (`p3_models.py`, `ner_models.py`)
- **LoRMA** (compulsory): output-side low-rank multiplicative-style update `h + s * B(A h)` on selected linear layers.
- **LoRA / rsLoRA**, **IA3**, **Houlsby / Pfeiffer adapters**, **BitFit**, **partial (last-k layer) fine-tuning**, **prompt tuning**, **full fine-tuning**.
- **Knowledge distillation**: logit KL for classification / summarization, sigmoid soft targets for multi-label, teacher similarity-score KL for retrieval. Student = half-depth copy (`--keep` layers) of a Phase 1 model.

## Results summary (Codabench test)
| Phase | Metric | Best submission | Score |
|---|---|---|---|
| 1 | BERTScore F1 | llama | 0.8340 |
| 2 | Strict macro-F1 | submission_2 | 0.3720 |
| 3 | Macro-F1 | p3_kd_llama6 | 0.4846 |
| 4 | Macro-F1 | p4_eff_full | 0.6112 |
| 5 | Macro-F1 | p5_llama_partial | 0.3077 |
| 6 | BERTScore F1 | p6_gpt2_houlsby | 0.8310 |
| 7 | Micro-F1@7 | Dense-ensemble + TF-IDF fusion | 0.3485 |

See each phase's README for the full experiment tables.

## Reproducibility notes
No model weights or data are included. Data is downloaded from the course Hugging Face dataset by the scripts. Random seeds are set by `--seed`.
