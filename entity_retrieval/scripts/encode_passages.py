"""Encode all ~21M DPR psgs_w100 passages with the DPR-NQ context encoder.
Saves fp16 embedding shards + aligned passage-id lists. Resumable by shard."""
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, "${WORKDIR}")
from dpr_common import load_tagged_encoders

TAG = os.environ.get("TAG", "nq")
SP = "${WORKDIR}"
TSV = f"{SP}/psgs_w100.tsv"
OUT = f"{SP}/dpr_{TAG}_passage_emb"; os.makedirs(OUT, exist_ok=True)
BATCH = 1024
SHARD = 1_000_000  # passages per output shard

dev = "cuda"
_q, enc, tok = load_tagged_encoders(TAG, device=dev, half=True)  # ctx encoder for TAG

@torch.no_grad()
def encode_batch(titles, texts):
    feats = tok(titles, texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
    return enc(**feats).last_hidden_state[:, 0].float().cpu().numpy().astype(np.float16)  # DPR [CLS]

def flush(emb_chunks, pids, shard_id):
    if os.path.exists(f"{OUT}/emb_{shard_id:03d}.npy"):
        print(f"  shard {shard_id} exists, skip", flush=True); return
    arr = np.concatenate(emb_chunks, axis=0).astype(np.float16)
    np.save(f"{OUT}/emb_{shard_id:03d}.npy", arr)
    json.dump(pids, open(f"{OUT}/pids_{shard_id:03d}.json", "w"))
    print(f"  wrote shard {shard_id}: {arr.shape}", flush=True)

t0 = time.time()
done = {int(f[4:7]) for f in os.listdir(OUT) if f.startswith('emb_')}
emb_chunks, pids = [], []          # accumulators for the CURRENT shard
shard_id, n, cur = 0, 0, 0         # cur = passages assigned to current shard
bt, bx, bp = [], [], []            # current encode batch

def run_batch():
    global emb_chunks, pids, bt, bx, bp
    if shard_id not in done:        # skip encoding for already-written shards (resume)
        emb_chunks.append(encode_batch(bt, bx)); pids.extend(bp)
    bt, bx, bp = [], [], []

with open(TSV) as f:
    f.readline()  # skip header 'id\ttext\ttitle'
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 3:
            continue
        text = p[1].strip()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        bp.append(p[0]); bx.append(text); bt.append(p[2].strip())
        if len(bp) == BATCH:
            run_batch(); n += BATCH; cur += BATCH
            if cur >= SHARD:                                  # flush full shard (~1M)
                if shard_id not in done:
                    flush(emb_chunks, pids, shard_id)
                emb_chunks, pids = [], []; cur = 0; shard_id += 1
            if n % 204800 == 0:
                el = time.time()-t0
                print(f"  {n:,} | {el:.0f}s | {n/el:.0f} p/s | ~{(21015324-n)/max(n/el,1)/60:.0f}min left", flush=True)
    if bp:
        run_batch(); n += len(bp); cur += len(bp)
    if cur > 0 and shard_id not in done:
        flush(emb_chunks, pids, shard_id)
print(f"DONE: {n:,} passages in {(time.time()-t0)/60:.1f} min", flush=True)
