# Phase 2: Legal named-entity recognition

Token-level tagging with 12 entity types (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO) using BIO-style tags; metric: strict macro-F1.

## Method
- `ner_data.py` loads the data from the course Hugging Face dataset and aligns character spans to BPE tokens.
- `ner_models.NERModel` puts a token-classification head on a Phase 1 backbone. Decoder-only backbones are made bidirectional at fine-tuning time (`--bidir 1`, `set_bidir`), which gives the tagger right-context.
- Adaptation: LoRMA (compulsory), LoRA, adapters, BitFit, partial fine-tuning, full fine-tuning, and knowledge distillation (`--teacher`, `--kd_alpha`). `--o_weight` down-weights the dominant `O` tag.
- `repredict.py` regenerates test predictions from a saved checkpoint.

## Commands
Exact commands of the six runs (notebook log; GPU 1, `M` = folder with the Phase 1 checkpoints):
```
python ner_train.py --name llama_lorma        --kind llama --base $M/llama_best.pt --tech lorma --epochs 15 --lr 5e-4
python ner_train.py --name gpt2_lora          --kind gpt2  --base $M/gpt2_best.pt  --tech lora  --epochs 15 --lr 5e-4
python ner_train.py --name llama_full         --kind llama --base $M/llama_best.pt --tech full  --epochs 10 --lr 5e-5
python ner_train.py --name eff_half_kd        --kind eff   --base $M/eff_best.pt   --tech full  --keep 1,3,5,7,9,11,13 --epochs 15 --lr 1e-4 --teacher ner_out/llama_full
python ner_train.py --name eff_half_nokd      --kind eff   --base $M/eff_best.pt   --tech full  --keep 1,3,5,7,9,11,13 --epochs 15 --lr 1e-4
python ner_train.py --name llama_lorma_causal --kind llama --base $M/llama_best.pt --tech lorma --epochs 15 --lr 5e-4 --bidir 0
python repredict.py <name>   # regenerates the test predictions (span clean-up fix) as ner_<name>_fixed.zip
```
Requires `HF_TOKEN` (task data is downloaded from the course dataset) and the Phase 1 tokenizer.

## Results (Codabench test, strict macro-F1)
| Submission | Model | Technique | Test |
|---|---|---|---|
| submission_2 | not recorded (early run, 8 Oct) | not recorded | **0.3720** |
| submission_3 | not recorded (early run, 8 Oct) | not recorded | 0.3640 |
| ner_eff_half_kd | eff, half depth student (7 of 14 layers) | full fine-tuning + knowledge distillation (teacher: llama_full) | 0.3420 |
| ner_eff_half_nokd | eff, half depth (7 of 14 layers) | full fine-tuning, without KD | 0.3246 |
| ner_llama_lorma_causal | llama | LoRMA, causal (unidirectional) attention | 0.3160 |
| ner_gpt2_lora | gpt2 | LoRA | 0.2145 |

Note: the test scores of `submission_2` and `submission_3` were not recorded against a configuration. The notebook shows six Phase 2 runs; four have a recorded test score (above), and the other two are `llama_lorma` (LoRMA, dev macro-F1 0.3768) and `llama_full` (full fine-tuning, dev 0.3277). The six valid Codabench submissions match these six runs, so these two runs most likely produced the 0.3720 and 0.3640 scores (in that order, following their dev scores), but we could not verify this mapping. The commands above are the exact ones of the notebook.

## Observations
- KD helps the half-depth `eff` student: 0.3246 -> 0.3420 (+0.017).
- Several earlier uploads (`submission_2`, `ner_gpt2_lora`) failed validation before a corrected zip was accepted.
- GPT-2 + LoRA is clearly the weakest (0.2145); the best runs (0.37) are well above it.
