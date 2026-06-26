"""Encode all test-set questions with the DPR-NQ question encoder ([CLS])."""
import os, sys, glob, json
import numpy as np, torch
sys.path.insert(0, "${WORKDIR}")
from dpr_common import load_tagged_encoders

TAG = os.environ.get("TAG", "nq")
SP = "${WORKDIR}"
TEST = f"{SP}/dataset/test"
q, _c, tok = load_tagged_encoders(TAG, device="cuda", half=True)

meta, texts = [], []
for path in sorted(glob.glob(f"{TEST}/*.test.json")):
    rel = os.path.basename(path).split(".")[0]
    for item in json.load(open(path)):
        meta.append({"relation": rel, "question": item["question"], "answers": item["answers"]})
        texts.append(item["question"])
print(f"{len(texts)} questions")

embs = []
with torch.no_grad():
    for i in range(0, len(texts), 512):
        f = tok(texts[i:i+512], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        embs.append(q(**f).last_hidden_state[:, 0].float().cpu().numpy().astype(np.float16))
Q = np.concatenate(embs, axis=0)
np.save(f"{SP}/dpr_{TAG}_question_emb.npy", Q)
json.dump(meta, open(f"{SP}/dpr_{TAG}_question_meta.json", "w"))
print("saved question embeddings", Q.shape)
