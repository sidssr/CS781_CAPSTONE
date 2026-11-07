# Phase 6: Legal judgment summarization

Generate a summary of a judgment; metric: BERTScore F1 (ROUGE-1/L secondary).

## Method (`p6_train.py`)
- Encoder-decoder summarizer built on the Phase 1 `encdec` model (`ed_model.EncDec`); decoder-only Phase 1 models (llama, gpt2, eff) read the source document followed by the summary as one sequence (source is the prefix, loss on the summary part).
- Adaptation: LoRMA, LoRA, adapters (Houlsby/Pfeiffer), IA3, partial FT, prompt tuning, full FT, and KD (`--teacher`: logit KL). `--keep_enc/--keep_dec` build half-depth students.
- Decoding: greedy (argmax) with n-gram blocking (`--ngram 3`), minimum/maximum new tokens (`--min_new 30`, `--max_new 160`).
- **Post-processing:** to address the length mismatch between generated and reference summaries, outputs of two or three models were merged with sentence-level deduplication and a word cap (420 or 600 words). This produced `merge2` (gpt2+Houlsby and llama+LoRA) and `merge3` (three models).

## Results (Codabench test, BERTScore F1)
| Submission | Model | Technique | Test |
|---|---|---|---|
| p6_gpt2_houlsby | gpt2 | Houlsby adapters | **0.8310** |
| p6_merge2 | gpt2 + llama outputs merged | post-processing (dedup, 420-word cap) | 0.8298 |
| p6_llama_lora | llama | LoRA | 0.8290 |
| p6_merge3 | 3 models merged | post-processing (dedup, 600-word cap) | 0.8288 |
| p6_eff_pfeiffer | eff | Pfeiffer adapters | 0.8283 |
| p6_enc_kd | encdec (half-depth student) | Knowledge distillation | 0.8232 |
| p6_enc_lorma | encdec | LoRMA | 0.8209 |
| p6_enc_prompt | encdec | Prompt tuning | 0.7977 |

## Observations
- All adapter/LoRA-style runs land within 0.003 BERTScore of each other (0.828-0.831); BERTScore F1 is compressive, so differences are small.
- Merging outputs of several models with sentence dedup and a length cap did not beat the best single model (0.8298 / 0.8288 vs 0.8310), i.e. fixing the length mismatch by post-processing gave no gain.
- Prompt tuning on the encoder-decoder is clearly worst (0.7977): only 20 soft tokens of capacity for a generation task.
- KD student (0.8232) and LoRMA (0.8209) on the same backbone differ by 0.002, which is within plausible noise (no clear difference).

## How to run
All scripts were run on Kaggle (T4 GPUs) and read their paths from environment variables with Kaggle defaults. Needed: the Phase 1 checkpoints (`llama_best.pt`, `gpt2_best.pt`, `eff_best.pt`, `encdec_best.pt`) in the folder given by `MB`, the BPE `tokenizer*.json`, the HuggingFace token in `HF_TOKEN` (the task data is downloaded from the course dataset `Exploration-Lab/CS-781-Capstone`), and the repository files copied to the working directory. Each run writes `<out>/best.pt`, `prediction.json` and a submission zip. Commands below use the script defaults unless a flag is shown; the exact per-run settings (learning rate, time budget) of the first runs were kept in `cfg.json` files that were lost with the Kaggle sessions, so a few runs may need those two settings adjusted (see Observations).

```
python p6_train.py --name p6_gpt2_houlsby --kind gpt2   --tech houlsby
python p6_train.py --name p6_llama_lora   --kind llama  --tech lora
python p6_train.py --name p6_eff_pfeiffer --kind eff    --tech pfeiffer
python p6_train.py --name p6_enc_lorma    --kind encdec --tech lorma
python p6_train.py --name p6_enc_prompt   --kind encdec --tech prompt --nprompt 20
python p6_train.py --name p6_enc_kd --kind encdec --tech lorma --keep_enc <layers> --keep_dec <layers> --teacher <teacher run dir>
```
The merged submissions (`p6_merge2`, `p6_merge3`) were produced by a separate post-processing step on saved predictions; no merge script is included in this folder.
