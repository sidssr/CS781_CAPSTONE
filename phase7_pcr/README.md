# Phase 7: Prior Case Retrieval (PCR)

Given a query case and a candidate pool, return 7 ranked candidate IDs per query (metric: Micro-F1@7). Train: 817 queries / 4,320 candidates; dev: 118 / 1,023; test: 237 / 1,727. Documents are ~9k tokens, so they are chunked.

## Method (`p7_train.py`)
- **Bi-encoder over Phase 1 backbones.** Each document is split into 256-token chunks. Each chunk is encoded by the (adapted) Phase 1 model, mean-pooled over tokens, projected (Linear to 256-d) and L2-normalised. The document vector is the renormalised mean of its chunk vectors. Training uses 4 random chunks per document; inference uses 24 evenly spread chunks.
- **Objective:** InfoNCE with temperature (learnable logit scale, init 1/0.05), in-batch negatives plus mined hard negatives (top-30 non-relevant candidates from the current model; re-mined at 50% of the time budget).
- **Retrieval:** cosine similarity, top-7.
- **Adaptation techniques:** LoRMA (llama), IA3 (gpt2), partial fine-tuning of last 4 layers (eff), prompt tuning (llama, 20 soft tokens), full fine-tuning (gpt2), and knowledge distillation (gpt2 half-depth student with LoRMA; teacher = llama+LoRMA; loss = 0.5 InfoNCE + 0.5 KL between student and teacher query-candidate similarity distributions).
- **Time-budgeted training:** 20 min per run with a cosine learning-rate schedule driven by wall-clock time.

## Extra lexical / fusion submissions (official data only)
- `fuse.py`: per-query z-scored dense scores averaged over models, fused with sublinear TF-IDF cosine; the weight is tuned on dev.
- `lex.py`: BM25 (unigram/bigram, k1, b grid) and BM25 + dense fusion.

## Commands
```
python p7_train.py --budget_min 20 --name llama_lorma --kind llama --tech lorma
python p7_train.py --budget_min 20 --name gpt2_ia3 --kind gpt2 --tech ia3 --lr 1e-3
python p7_train.py --budget_min 20 --name eff_partial --kind eff --tech partial --last_k 4
python p7_train.py --budget_min 20 --name llama_prompt --kind llama --tech prompt
python p7_train.py --budget_min 20 --name gpt2_full --kind gpt2 --tech full --lr 5e-5 --head_lr 1e-3 --bq 3
python p7_train.py --budget_min 20 --name gpt2_kd --kind gpt2 --tech lorma --keep 0,1,2,3,4,5 --teacher p7_out/llama_lorma
python fuse.py llama_lorma gpt2_ia3 gpt2_kd eff_partial llama_prompt
python lex.py llama_lorma gpt2_ia3 gpt2_kd eff_partial llama_prompt
```

## Results
| Submission | Model | Technique | Dev F1@7 | Test F1@7 |
|---|---|---|---|---|
| llama_lorma | llama | LoRMA | 0.2546 | 0.2286 |
| gpt2_ia3 | gpt2 | IA3 | 0.2435 | 0.2315 |
| eff_partial | eff | Partial FT | 0.2223 | 0.1949 |
| llama_prompt | llama | Prompt tuning | 0.2513 | 0.2310 |
| gpt2_kd | gpt2 (6 layers) | LoRMA + KD | 0.2635 | 0.2369 |
| gpt2_full (rerun) | gpt2 | Full FT | 0.2412 | 0.2174 |
| gpt2_full (first run) | gpt2 | Full FT | not recorded | 0.2162 |
| fusion | 5 dense models + TF-IDF | score fusion | 0.3502 | **0.3485** |
| bm25 | none | BM25 | not recorded | 0.3355 |
| bm25_fusion | dense + BM25 | score fusion | not recorded | 0.3408 |

The dev score of llama_lorma is 0.2546 in `fuse.py`'s re-evaluation (the trainer's own log printed 0.2568). TF-IDF alone reaches 0.3391 on dev, higher than any adapted LM.

## Analysis points
- Adaptation barely changes retrieval quality: zero-shot dev F1 is ~0.26 for llama, and fine-tuned runs stay at 0.22-0.26.
- Lexical matching dominates: Phase 1 mean-pooled chunk embeddings transfer poorly to citation-style case relevance; fusion with TF-IDF gives about +9 points on dev.
- A high learning rate (1e-3) for LoRMA damaged the model (dev 0.17-0.19).
- More inference chunks help monotonically (4 chunks ~0.21 -> 24 chunks ~0.25 dev).
- About 21 of 118 dev queries have no relevant candidate in the dense top-7.
- Memory: K=8 chunks with 4 queries x 4 candidates exceeds a 15 GB T4; use smaller batch with gradient accumulation.
