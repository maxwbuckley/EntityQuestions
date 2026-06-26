"""Exact GPU top-100 dense retrieval over the 21M passage embedding shards.
Avoids a 65GB-RAM flat FAISS index by streaming shards through the GPU and
maintaining a running top-100 per question (inner-product, matching DPR)."""
import os, sys, glob, json, time
import numpy as np, torch

TAG = os.environ.get("TAG", "nq")
SP = "${WORKDIR}"
EMB = f"{SP}/dpr_{TAG}_passage_emb"
K = 100
QB = 2048  # query tile

dev = "cuda"
Q = torch.from_numpy(np.load(f"{SP}/dpr_{TAG}_question_emb.npy")).to(dev).half()  # (Nq,768)
Nq = Q.shape[0]
print(f"queries: {Q.shape}")

shard_ids = sorted(int(f[4:7]) for f in os.listdir(EMB) if f.startswith("emb_"))
# running global top-K
best_scores = torch.full((Nq, K), float("-inf"), device=dev, dtype=torch.float16)
best_gidx = torch.full((Nq, K), -1, device=dev, dtype=torch.int64)

offset = 0
all_pids = []
t0 = time.time()
for sid in shard_ids:
    emb = torch.from_numpy(np.load(f"{EMB}/emb_{sid:03d}.npy")).to(dev).half()  # (ns,768)
    pids = json.load(open(f"{EMB}/pids_{sid:03d}.json"))
    all_pids.extend(pids)
    ns = emb.shape[0]
    embT = emb.t().contiguous()  # (768, ns)
    for qs in range(0, Nq, QB):
        qe = Q[qs:qs+QB]                       # (qb,768)
        scores = qe @ embT                     # (qb, ns) fp16
        v, i = scores.topk(K, dim=1)           # (qb,K)
        gi = i + offset
        cs = torch.cat([best_scores[qs:qs+QB], v], dim=1)       # (qb,2K)
        cg = torch.cat([best_gidx[qs:qs+QB], gi], dim=1)
        nv, ni = cs.topk(K, dim=1)
        best_scores[qs:qs+QB] = nv
        best_gidx[qs:qs+QB] = torch.gather(cg, 1, ni)
    offset += ns
    print(f"  shard {sid}: ns={ns} total={offset} ({time.time()-t0:.0f}s)", flush=True)

gidx = best_gidx.cpu().numpy()  # (Nq,K) global passage row indices
all_pids = np.array(all_pids)   # row index -> pid string
topk_pids = all_pids[gidx]      # (Nq,K) pid strings
np.save(f"{SP}/dpr_{TAG}_topk_pids.npy", topk_pids)
np.save(f"{SP}/dpr_{TAG}_topk_scores.npy", best_scores.cpu().numpy())
print(f"DONE search in {(time.time()-t0)/60:.1f} min; topk_pids {topk_pids.shape}")
