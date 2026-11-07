# Phase 5: Legal statute identification (multi-label)

Predict the set of statutes applicable to a case; metric: macro-F1 (also micro-F1 and subset accuracy).

## Method (`p5_train.py`)
- Same long-document pipeline as Phase 4 (head+tail text, 256-token chunks, chunk selection, attention pooling) with a multi-label sigmoid head.
- Loss: binary cross-entropy with `pos_weight` ~ (neg/pos)^0.5 (`--pw_pow`), or focal loss (`--loss focal --gamma 2`). Per-label thresholds are tuned on dev.
- Adaptation: LoRMA, LoRA, adapters, IA3, BitFit, partial FT, prompt tuning (`--nprompt`), full FT, and KD with sigmoid soft targets from a teacher (`--teacher`).

## Results (Codabench test, macro-F1)
| Submission | Model | Technique | Test |
|---|---|---|---|
| p5_llama_partial | llama | Partial FT (last layers) | **0.3077** |
| p5_llama_kd | llama, half-depth student | Knowledge distillation (sigmoid soft targets) | 0.2845 |
| p5_llama_lorma | llama | LoRMA | 0.2817 |
| p5_eff_adapter | eff | Bottleneck adapters | 0.2811 |
| p5_gpt2_prompt | gpt2 | Prompt tuning | 0.2534 |
| p5_enc_full | enc (encdec encoder) | Full fine-tuning | 0.2359 |

## Observations
- Partial fine-tuning again wins (0.3077), as in Phase 4; LoRMA, adapters and the KD student cluster at 0.28.
- Absolute macro-F1 is low for every method (100 statutes, about four per document), and the differences between methods are only a few points; we did not isolate why scores are low.
- Prompt tuning (0.2534) and the encoder-only model with full FT (0.2359) are the weakest.

## How to run
All scripts were run on Kaggle (T4 GPUs) and read their paths from environment variables with Kaggle defaults. Needed: the Phase 1 checkpoints (`llama_best.pt`, `gpt2_best.pt`, `eff_best.pt`, `encdec_best.pt`) in the folder given by `MB`, the BPE `tokenizer*.json`, the HuggingFace token in `HF_TOKEN` (the task data is downloaded from the course dataset `Exploration-Lab/CS-781-Capstone`), and the repository files copied to the working directory. Each run writes `<out>/best.pt`, `prediction.json` and a submission zip. Commands below use the script defaults unless a flag is shown; the exact per-run settings (learning rate, time budget) of the first runs were kept in `cfg.json` files that were lost with the Kaggle sessions, so a few runs may need those two settings adjusted (see Observations).

```
python p5_train.py --name p5_llama_partial --kind llama --tech partial
python p5_train.py --name p5_llama_lorma   --kind llama --tech lorma
python p5_train.py --name p5_eff_adapter   --kind eff   --tech adapter
python p5_train.py --name p5_gpt2_prompt   --kind gpt2  --tech prompt --nprompt 20
python p5_train.py --name p5_enc_full      --kind enc   --tech full
python p5_train.py --name p5_llama_kd --kind llama --tech lorma --keep <layers> --teacher <teacher run dir>
```
