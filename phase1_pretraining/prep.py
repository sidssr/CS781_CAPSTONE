import os, json, random, numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer

tok_dir = "/kaggle/working/tok32k"
data_dir = "/kaggle/working/data"
os.makedirs(tok_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

path = hf_hub_download("Exploration-Lab/CS-781-Capstone",
                       "Phase_1_Pretraining/Train/train_supreme_court.json",
                       repo_type="dataset", token=os.environ["HF_TOKEN"])
records = json.load(open(path, encoding="utf-8"))
print("docs:", len(records), flush=True)

random.seed(42)
idx = list(range(len(records)))
random.shuffle(idx)

tok = ByteLevelBPETokenizer()
tok.train_from_iterator((records[i]["text"] for i in idx[:5000]),
                        vocab_size=32000, min_frequency=2,
                        special_tokens=["<pad>", "<eos>"])
tok.save_model(tok_dir)
tok.save(tok_dir + "/tokenizer.json")
EOS = tok.token_to_id("<eos>")
print("tokenizer done, EOS id:", EOS, flush=True)

TRAIN_TARGET, VAL_TARGET = 120_000_000, 3_000_000
train_chunks, val_chunks = [], []
n_train = n_val = 0
for s in range(0, len(idx), 500):
    ids = idx[s:s + 500]
    encs = tok.encode_batch([records[i]["text"] for i in ids])
    for k, e in enumerate(encs):
        arr = np.array(e.ids + [EOS], dtype=np.uint16)
        if (s + k) % 50 == 0:
            if n_val < VAL_TARGET:
                val_chunks.append(arr); n_val += len(arr)
        elif n_train < TRAIN_TARGET:
            train_chunks.append(arr); n_train += len(arr)
    print(f"tokens: train {n_train:,} val {n_val:,}", flush=True)
    if n_train >= TRAIN_TARGET and n_val >= VAL_TARGET:
        break

np.concatenate(train_chunks).tofile(data_dir + "/train.bin")
np.concatenate(val_chunks).tofile(data_dir + "/val.bin")
print("PREP DONE", n_train, n_val, flush=True)