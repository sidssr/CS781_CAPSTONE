# Phase 1: Legal language model pretraining

Four models are pretrained from scratch on the official Supreme Court judgment corpus (`prep.py` trains a 32k byte-level BPE tokenizer and writes uint16 token shards, ~120M train tokens, 3M val tokens).

| Model | Script | Architecture |
|---|---|---|
| llama | `lm.py` (`--arch llama`) | RMSNorm, RoPE, SwiGLU, no biases, causal attention; default d=768, 12 layers, 12 heads, ctx 1024 |
| gpt2 | `lm.py` (`--arch gpt2`) | LayerNorm, learned positions, GELU MLP; same default size |
| eff | `eff.py` / `eff_model.py` | Sliding-window local attention (window 256) with a global layer every 4th block, grouped-query attention (12 heads, 4 KV heads), 14 layers, d=768, ctx 1024 |
| encdec | `encdec.py` / `ed_lib.py`, `ed_model.py` | Encoder-decoder, 5 encoder + 5 decoder layers, d=768, RoPE, cross-attention; encoder ctx 768, decoder ctx 256; trained as a continuation model: the encoder reads the first E tokens of a window and the decoder predicts the next D tokens (`ed_lib.get_batch`) |

Training: AdamW, warmup + cosine decay, bf16/fp16 autocast, time-limited runs on a Kaggle T4. `gen.py` / `gen_run.py` generate continuations for the Phase 1 evaluation (BERTScore F1 primary, ROUGE-L and BLEU-4 secondary). `eff_model.generate` uses repetition penalty (1.1) and n-gram blocking (4).

## Relation to the course starter notebook
The starter notebook trains a Hugging Face `GPT2LMHeadModel` (8 layers, d=560, ctx 2048, 200M tokens, GPT-2 tokenizer) and generates 128 words per test prompt (temperature 0.1, top-p 0.95, up to 384 new tokens). Our pipeline differs: all four architectures are implemented from scratch, with our own 32k byte-level BPE tokenizer trained on the corpus; generation is greedy (temperature 0) with repetition penalty 1.1 and 4-gram blocking, up to 220 new tokens, truncated to the first 128 words (`gen.py`, `gen_run.py`).

## Pretraining data and budget
Supreme Court judgments only (`train_supreme_court.json`, 55,549 documents, mean ~5.8k words, median ~3.5k). Tokenizer trained on 5,000 randomly chosen documents (vocab 32,000, special tokens `<pad>`, `<eos>`). Token budget: 120M train / 3M validation tokens (every 50th document held out for validation), stored as uint16 `train.bin` / `val.bin`.

| Model | Params | Tokens trained | Throughput (T4) | Best val loss (ppl) |
|---|---|---|---|---|
| llama | 109.5M | 100M | (see note) | 2.7204 (15.19) |
| gpt2 | 110.3M | 100M | (see note) | 2.7931 (16.33) |
| eff | 112.7M | 100M | ~13k tok/s (~2.1 h) | 2.7195 (15.17) |
| encdec | 107.2M | 200M | ~29-31k tok/s (~1.8 h) | 3.0460 (21.03) |

(llama and gpt2 validation losses were re-evaluated from the saved best checkpoints on the full held-out set of 3,006,376 tokens; eff and encdec values are training-time estimates. encdec validation loss is for continuation of a 256-token decoder window given 768 encoder tokens, so it is not directly comparable to the decoder-only losses.)

## Results (Codabench test, BERTScore F1)
| Model | BERTScore F1 |
|---|---|
| llama | **0.8340** |
| eff | 0.8338 |
| gpt2 | 0.8334 |
| encdec | 0.8236 |

An earlier first submission (`submission_1`, 30 Sep) scored 0.8007. The first encdec upload failed/was cancelled and was resubmitted as `submission_encdec_1`. The four final models are within 0.01 BERTScore of each other, so the choice of architecture matters little for this metric.

## Notes
The original encoder-decoder checkpoint (val loss 3.046, the one behind the 0.8236 Phase 1 score) was later lost with a Kaggle session reset, so for Phases 4-7 the encoder was retrained with the same scripts (val loss 3.59); its encoder is later loaded as an encoder-only backbone (`ner_models.EncOnly`).

## How to run
All scripts were run on Kaggle (T4 GPUs) and read their paths from environment variables with Kaggle defaults. Needed: the HuggingFace token in `HF_TOKEN` (the task data is downloaded from the course dataset `Exploration-Lab/CS-781-Capstone`), and the repository files copied to the working directory. Each run writes `<out>/best.pt`, `prediction.json` and a submission zip. Commands below use the script defaults unless a flag is shown; the exact per-run settings (learning rate, time budget) of the first runs were kept in `cfg.json` files that were lost with the Kaggle sessions, so a few runs may need those two settings adjusted (see Observations).

```
python prep.py                                   # tokenizer + token shards (needs HF_TOKEN)
python lm.py   --arch llama --name llama         # llama-style, defaults: d=768, 12 layers, 100M tokens
python lm.py   --arch gpt2  --name gpt2
python eff.py  --name eff
python encdec.py --name encdec
python valloss.py                                # re-evaluate llama/gpt2 on the rebuilt validation set
python gen_run.py <name>                         # generate continuations for the Phase 1 submission
```
