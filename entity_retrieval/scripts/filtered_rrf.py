"""RRF fusion of (A) name-filtered DPR over the diacritic-folded postings index and
(B) global exact DPR nearest neighbors, on the GLiNER-large person subset.
Weighted RRF gives the exact-match list a tiny bonus to break ties in its favor."""
import os, sys, json, time
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
import numpy as np, torch
REPO = "${REPO}"
SP = "${WORKDIR}"
sys.path.insert(0, REPO); sys.path.insert(0, SP)
from pyserini.search.lucene import LuceneSearcher
import utils.ion as ion
from utils.has_answer_fn import string_match
from utils.tokenizers import SimpleTokenizer
from fold_util import fold

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = sys.argv[2] if len(sys.argv) > 2 else "nq"   # nq | ft : which DPR embeddings
K_NN = int(os.environ.get("K_NN", 1000))      # global nearest neighbors (list B)
K_CAND = int(os.environ.get("K_CAND", 1000))  # name-filter candidates cap (list A)
RRF_C = 60
EXACT_BONUS = 1e-4   # tiny additive weight so ties break toward the exact-match list
K_VALUES = [1, 5, 20, 100]
SHARD = 1000448
tok = SimpleTokenizer()

meta = json.load(open(f"{SP}/dpr_nq_question_meta.json"))
gl = json.load(open(f"{SP}/gliner_results_large.json"))
Q = np.load(f"{SP}/dpr_{TAG}_question_emb.npy").astype(np.float32)
dpr_topk = np.load(f"{SP}/dpr_{TAG}_topk_pids.npy", allow_pickle=True)   # top-100 (for DPR-full eval)
EMB_DIR = f"{SP}/dpr_{TAG}_passage_emb"
pid2title = ion.read_json(f"{SP}/pid2title.json")
s_fold = LuceneSearcher(f"{SP}/bm25_index_folded"); s_fold.set_bm25(0.9, 0.4)
s_orig = LuceneSearcher(f"{SP}/bm25_index")
shards = [np.load(f"{EMB_DIR}/emb_{i:03d}.npy", mmap_mode="r") for i in range(22)]

def emb_of(pid):
    r = int(pid) - 1
    return np.asarray(shards[r // SHARD][r % SHARD], dtype=np.float32)

_body = {}
def body_of(pid):
    pid = str(pid)
    if pid not in _body:
        d = s_orig.doc(pid)
        c = json.loads(d.raw())["contents"] if d else ""
        t = pid2title.get(pid, "")
        _body[pid] = c[len(t):].strip()
    return _body[pid]

_fc = {}
def folded_contents(pid):
    pid = str(pid)
    if pid not in _fc:
        d = s_fold.doc(pid)
        _fc[pid] = json.loads(d.raw())["contents"] if d else ""
    return _fc[pid]

def found_rank(pids, answers):
    for rank, pid in enumerate(pids):
        if string_match({"text": body_of(pid)}, answers, tok):
            return rank
    return -1

person_idx = [i for i, g in enumerate(gl) if g["has_person"]]
if LIMIT:                       # LIMIT=0 -> all person-questions
    person_idx = person_idx[:LIMIT]
print(f"person-questions: {len(person_idx)}")

# ---- list B: global exact top-K_NN NN for these queries, on GPU ----
dev = "cuda"
Qsub = torch.from_numpy(Q[person_idx]).to(dev).half()       # (N,768)
N = Qsub.shape[0]
best = torch.full((N, K_NN), float("-inf"), device=dev, dtype=torch.float16)
bidx = torch.full((N, K_NN), -1, device=dev, dtype=torch.int64)
offset = 0; all_pids = []; t0 = time.time()
QB = int(os.environ.get("QB", 1024))                        # tile queries to bound GPU memory
for i in range(22):
    emb = torch.from_numpy(np.load(f"{EMB_DIR}/emb_{i:03d}.npy")).to(dev).half()
    pids_i = json.load(open(f"{EMB_DIR}/pids_{i:03d}.json"))
    embT = emb.t().contiguous()
    for qs in range(0, N, QB):                              # (qb, ns) score tile, not (N, ns)
        sc = Qsub[qs:qs+QB] @ embT
        v, idx = sc.topk(min(K_NN, sc.shape[1]), dim=1)
        cs = torch.cat([best[qs:qs+QB], v], 1); cg = torch.cat([bidx[qs:qs+QB], idx + offset], 1)
        nv, ni = cs.topk(K_NN, dim=1)
        best[qs:qs+QB] = nv; bidx[qs:qs+QB] = torch.gather(cg, 1, ni)
    all_pids.extend(pids_i); offset += len(pids_i)
    del emb, embT
all_pids = np.array(all_pids)
nn_pids = all_pids[bidx.cpu().numpy()]                       # (N, K_NN) pid strings
print(f"global NN computed in {time.time()-t0:.0f}s")

# BM25 ranks (existing results)
import glob
bm25_rank = {}
for f in glob.glob(f"{SP}/bm25_results/*.test.json"):
    for r in ion.read_json(f):
        fr = -1
        for j, c in enumerate(r["ctxs"]):
            if c["has_answer"]: fr = j; break
        bm25_rank[r["question"]] = fr

def rrf_fuse(listA, listB, exact_bonus):
    from collections import defaultdict
    sc = defaultdict(float)
    setA = set(listA)
    for rank, pid in enumerate(listA): sc[pid] += 1.0 / (RRF_C + rank + 1)
    for rank, pid in enumerate(listB): sc[pid] += 1.0 / (RRF_C + rank + 1)
    for pid in setA: sc[pid] += exact_bonus           # tiny edge to exact match
    return sorted(sc, key=lambda p: -sc[p])

from collections import defaultdict
hit = {s: defaultdict(int) for s in ("bm25","dpr_full","dpr_filt","rrf","rrf_bonus")}
zero_cand = 0; cand_sizes = []
t0 = time.time()
for n, qi in enumerate(person_idx):
    answers = meta[qi]["answers"]; q = Q[qi]
    # BM25
    fr = bm25_rank.get(meta[qi]["question"], -1)
    for k in K_VALUES:
        if 0 <= fr < k: hit["bm25"][k] += 1
    # DPR full
    fr = found_rank(list(dpr_topk[qi]), answers)
    for k in K_VALUES:
        if 0 <= fr < k: hit["dpr_full"][k] += 1
    # list A: name-filtered (folded index) ranked by DPR
    cand = []; seen = set()
    for name in gl[qi]["persons"]:
        fn = fold(name).lower()
        try: hits = s_fold.search(f'"{fold(name)}"', k=K_CAND)
        except Exception: hits = []
        for h in hits:
            if h.docid in seen: continue
            if fn in folded_contents(h.docid).lower():
                seen.add(h.docid); cand.append(h.docid)
    cand_sizes.append(len(cand))
    if not cand: zero_cand += 1
    listA = []
    if cand:
        C = np.stack([emb_of(p) for p in cand])
        listA = [cand[o] for o in np.argsort(-(C @ q))]
        fr = found_rank(listA, answers)
        for k in K_VALUES:
            if 0 <= fr < k: hit["dpr_filt"][k] += 1
    # list B: global NN
    listB = [str(p) for p in nn_pids[n]]
    # RRF fusions
    fused = rrf_fuse(listA, listB, 0.0)
    fr = found_rank(fused, answers)
    for k in K_VALUES:
        if 0 <= fr < k: hit["rrf"][k] += 1
    fused_b = rrf_fuse(listA, listB, EXACT_BONUS)
    fr = found_rank(fused_b, answers)
    for k in K_VALUES:
        if 0 <= fr < k: hit["rrf_bonus"][k] += 1
    if (n+1) % 100 == 0:
        print(f"  {n+1}/{len(person_idx)} ({time.time()-t0:.0f}s)", flush=True)

M = len(person_idx)
print(f"\n=== Person subset, N={M}  DPR={TAG.upper()}  (folded index, RRF K_NN={K_NN}, exact_bonus={EXACT_BONUS}) ===")
print(f"{'system':18} " + " ".join(f"top{k:>3}" for k in K_VALUES))
for s, label in [("bm25","BM25"),("dpr_full",f"DPR-{TAG} full"),("dpr_filt",f"DPR-{TAG} name-filt"),
                 ("rrf","RRF fuse"),("rrf_bonus","RRF + exact-bonus")]:
    print(f"{label:16} " + " ".join(f"{100*hit[s][k]/M:5.1f}" for k in K_VALUES))
print(f"\nzero-candidate questions (folded): {zero_cand}/{M} ({100*zero_cand/M:.1f}%), "
      f"avg cand {np.mean(cand_sizes):.1f}")
print(f"done in {(time.time()-t0)/60:.1f} min")
out = {"tag": TAG, "N": M, "zero_cand": zero_cand, "avg_cand": float(np.mean(cand_sizes)),
       "systems": {s: {k: 100*hit[s][k]/M for k in K_VALUES} for s in hit}}
json.dump(out, open(f"{SP}/rrf_{TAG}_full_results.json", "w"), indent=2)
