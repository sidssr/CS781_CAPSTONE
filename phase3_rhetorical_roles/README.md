# Phase 3: Rhetorical role labelling

Each judgment is a sequence of sentences, each assigned a rhetorical role; metric: macro-F1.

## Method (`p3_train.py`, `p3_models.py`)
- `SentEnc` encodes each sentence with a Phase 1 backbone (optionally with soft prompts) and returns mean-pooled and last-token states.
- `RRModel` projects sentence vectors, adds a domain embedding and a relative-position feature (sin/cos of position in the document), and runs a bidirectional GRU over the sentence sequence (`--ctx 1`) so each sentence sees its neighbours. An optional linear-chain CRF (`--crf 1`) is available.
- Loss: class-weighted cross-entropy with label smoothing (`--cw_pow`, `--ls`) plus an auxiliary per-sentence loss (`--aux`). Knowledge distillation from a teacher via `--teacher`.
- Adaptation: LoRMA, LoRA, IA3, adapters, BitFit, partial FT, prompt tuning, full FT.
- `ens.py` ensembles several runs' probabilities (weights selected on dev) and writes the final predictions.

## Results (Codabench test, macro-F1)
| Submission | Model | Technique | Test |
|---|---|---|---|
| p3_kd_llama6 | llama, 6-layer student (layers 0,2,4,6,8,10) | full fine-tuning + knowledge distillation (teacher: a full-depth llama run) | **0.4846** |
| p3_gpt2_lora | gpt2 | LoRA | 0.4523 |
| p3_llama_lorma | llama | LoRMA | 0.4521 |
| p3_eff_rslora | eff | rsLoRA | 0.4406 |
| p3_gpt2_prompt | gpt2 | Prompt tuning | 0.4268 |
| p3_llama_ia3 | llama | IA3 | 0.4112 |

## Observations
- The distilled 6-layer llama student (0.4846) beats every full-depth PEFT run, including the LoRMA run on the same backbone (0.4521, +0.033) while having half the layers. The student is fully fine-tuned and there is no non-distilled 6-layer control, so the gain cannot be attributed to distillation alone.
- LoRA-family methods (LoRA, LoRMA, rsLoRA) cluster at 0.44-0.45; IA3 and prompt tuning are lower (0.41-0.43), consistent with them having far fewer trainable parameters.

## How to run
All scripts were run on Kaggle (T4 GPUs) and read their paths from environment variables with Kaggle defaults. Needed: the Phase 1 checkpoints (`llama_best.pt`, `gpt2_best.pt`, `eff_best.pt`, `encdec_best.pt`) in the folder given by `MB`, the BPE `tokenizer*.json`, the HuggingFace token in `HF_TOKEN` (the task data is downloaded from the course dataset `Exploration-Lab/CS-781-Capstone`), and the repository files copied to the working directory. Each run writes `<out>/best.pt`, `prediction.json` and a submission zip. Commands below use the script defaults unless a flag is shown; the exact per-run settings (learning rate, time budget) of the first runs were kept in `cfg.json` files that were lost with the Kaggle sessions, so a few runs may need those two settings adjusted (see Observations).

```
# commands of the notebook runs (the .zip of each run is p3_<name>.zip)
python p3_train.py --epochs 8  --name llama_lorma --kind llama --tech lorma --r 16 --alpha 32 --targets qkv,o --lr 1e-3
python p3_train.py --epochs 16 --name gpt2_lora   --kind gpt2  --tech lora  --r 16 --alpha 32 --targets qkv,o --lr 1e-3
python p3_train.py --epochs 16 --name eff_rslora  --kind eff   --tech rslora --r 16 --alpha 8 --targets q,kv,o --lr 1e-3
python p3_train.py --epochs 16 --name gpt2_prompt --kind gpt2  --tech prompt --nprompt 20 --prompt_lr 1e-2
python p3_train.py --epochs 16 --name llama_ia3   --kind llama --tech ia3 --targets qkv,w2 --lr 3e-3
# distilled half-depth student (llama layers 0,2,4,6,8,10, fully fine-tuned)
python p3_train.py --epochs 12 --name kd_llama6 --kind llama --tech full --keep 0,2,4,6,8,10 --lr 1e-4 --teacher p3_out/llama_lorma
python p3_train.py --epochs 10 --name llama_full --kind llama --tech full --lr 6e-5     # full-depth teacher used in the last launch
# (the last launch used --teacher p3_out/llama_full; which teacher gave the submitted 0.4846 was not recorded)
python ens.py        # optional: ensemble run probabilities (weights chosen on dev)
```
