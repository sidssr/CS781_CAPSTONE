# Phase 4: Court judgment prediction and explanation (binary outcome)

Predict the judgment outcome of a complete case; metric: macro-F1.

## Method (`p4_train.py`)
- Cases are very long, so each is reduced to head+tail text (`--head_chars`, `--tail_chars`), split into 256-token chunks, and a subset of chunks is selected (`--sel firstlast | spread | random`).
- Each chunk goes through a Phase 1 backbone; chunk vectors are combined with attention pooling and fed to a classification head.
- Class imbalance: class-weighted cross-entropy (`--cw_pow`) with label smoothing; the decision threshold is tuned on dev.
- Adaptation: LoRMA, LoRA, adapters (Houlsby/Pfeiffer), IA3, BitFit, partial FT, full FT, and KD (`--teacher`: student learns from the teacher's logits with KL). Time-budgeted training (`--budget_min`).
- All four Phase 1 models are used (`--kind llama|gpt2|eff|enc`; `enc` loads the encdec encoder).

## Results (Codabench test, macro-F1)
| Submission | Model | Technique | Test |
|---|---|---|---|
| p4_eff_full | eff | Full fine-tuning | **0.6112** |
| p4_llama_partial | llama | Partial FT (last layers) | 0.6084 |
| p4_eff_adapter | eff | Bottleneck adapters | 0.5761 |
| p4_gpt2_bitfit | gpt2 | BitFit | 0.5746 |
| p4_llama_kd | llama, half-depth student | Knowledge distillation | 0.5720 |
| p4_llama_lorma_1 | llama | LoRMA (lr 3e-4, corrected) | 0.5624 |
| p4_enc_full | enc (encdec encoder) | Full fine-tuning | 0.5493 |
| p4_llama_lorma | llama | LoRMA (first run, lr 1e-3) | 0.3786 |

## Observations
- All four Phase 1 backbones are covered (llama, gpt2, eff, encoder).
- Full/partial fine-tuning (0.61) beat the parameter-efficient methods (0.56-0.58) on this long-document binary task.
- LoRMA with lr 1e-3 stalled (macro-F1 0.3786, near chance); lowering the LoRMA learning rate to 3e-4 with a higher head learning rate fixed it (0.5624).
- Several uploads failed validation and were resubmitted (`p4_gpt2_bitfit`, `p4_llama_lorma_1`).
- The encoder-only model (0.5493) is the weakest of the full-FT runs; we did not test why.

## How to run
All scripts were run on Kaggle (T4 GPUs) and read their paths from environment variables with Kaggle defaults. Needed: the Phase 1 checkpoints (`llama_best.pt`, `gpt2_best.pt`, `eff_best.pt`, `encdec_best.pt`) in the folder given by `MB`, the BPE `tokenizer*.json`, the HuggingFace token in `HF_TOKEN` (the task data is downloaded from the course dataset `Exploration-Lab/CS-781-Capstone`), and the repository files copied to the working directory. Each run writes `<out>/best.pt`, `prediction.json` and a submission zip. Commands below use the script defaults unless a flag is shown; the exact per-run settings (learning rate, time budget) of the first runs were kept in `cfg.json` files that were lost with the Kaggle sessions, so a few runs may need those two settings adjusted (see Observations).

```
python p4_train.py --name p4_eff_full      --kind eff   --tech full
python p4_train.py --name p4_llama_partial --kind llama --tech partial
python p4_train.py --name p4_eff_adapter   --kind eff   --tech adapter
python p4_train.py --name p4_gpt2_bitfit   --kind gpt2  --tech bitfit
python p4_train.py --name p4_enc_full      --kind enc   --tech full
python p4_train.py --name p4_llama_lorma_1 --kind llama --tech lorma --lr 3e-4 --head_lr 2e-3
python p4_train.py --name p4_llama_kd --kind llama --tech lorma --keep <layers> --teacher <teacher run dir>
```
