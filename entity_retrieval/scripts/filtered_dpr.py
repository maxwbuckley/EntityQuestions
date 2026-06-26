"""Entity-constrained dense retrieval on the GLiNER-large person subset.
Compares, over questions naming >=1 person:
  (A) BM25            (existing results)
  (B) DPR-NQ full     (dense over all 21M passages)
  (C) DPR-NQ filtered: hard-filter to passages whose contents contain the person
      name (exact substring, found via the positional inverted index / phrase query),
      then rank ONLY those by DPR vector similarity.
"""
import os, sys, json, time
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
import numpy as np
REPO = "${REPO}"
SP = "${WORKDIR}"
sys.path.insert(0, REPO)
from pyserini.search.lucene import LuceneSearcher
import utils.ion as ion
from utils.has_answer_fn import string_match
from utils.tokenizers import SimpleTokenizer

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = all person-questions
KCAND = 300                                            # phrase-query candidate cap per name
K_VALUES = [1, 5, 20, 100]
SHARD = 1000448
tok = SimpleTokenizer()

meta = json.load(open(f"{SP}/dpr_nq_question_meta.json"))
gl = json.load(open(f"{SP}/gliner_results_large.json"))
Q = np.load(f"{SP}/dpr_nq_question_emb.npy").astype(np.float32)        # (Nq,768)
dpr_topk = np.load(f"{SP}/dpr_nq_topk_pids.npy", allow_pickle=True)    # (Nq,100)
pid2title = ion.read_json(f"{SP}/pid2title.json")
searcher = LuceneSearcher(f"{SP}/bm25_index"); searcher.set_bm25(0.9, 0.4)
shards = [np.load(f"{SP}/dpr_nq_passage_emb/emb_{i:03d}.npy", mmap_mode="r") for i in range(22)]

def emb_of(pid):
    r = int(pid) - 1
    return np.asarray(shards[r // SHARD][r % SHARD], dtype=np.float32)

_text = {}
def contents_of(pid):
    pid = str(pid)
    if pid not in _text:
        d = searcher.doc(pid)
        _text[pid] = json.loads(d.raw())["contents"] if d else ""
    return _text[pid]

def body_of(pid):
    title = pid2title.get(str(pid), "")
    return contents_of(pid)[len(title):].strip()

def found_rank(pids, answers):
    for rank, pid in enumerate(pids):
        if string_match({"text": body_of(pid)}, answers, tok):
            return rank
    return -1

# Pre-load BM25 person-subset has_answer from existing results (keyed by question)
bm25_rank = {}
import glob
for f in glob.glob(f"{SP}/bm25_results/*.test.json"):
    for r in ion.read_json(f):
        fr = -1
        for i, c in enumerate(r["ctxs"]):
            if c["has_answer"]: fr = i; break
        bm25_rank[r["question"]] = fr

person_idx = [i for i, g in enumerate(gl) if g["has_person"]]
if LIMIT: person_idx = person_idx[:LIMIT]
print(f"person-questions: {len(person_idx)}")

from collections import defaultdict
hit = {s: defaultdict(int) for s in ("bm25", "dpr_full", "dpr_filt")}
cand_sizes = []
t0 = time.time()
for n, qi in enumerate(person_idx):
    answers = meta[qi]["answers"]; q = Q[qi]
    # (A) BM25
    fr = bm25_rank.get(meta[qi]["question"], -1)
    for k in K_VALUES:
        if 0 <= fr < k: hit["bm25"][k] += 1
    # (B) DPR full
    fr = found_rank(list(dpr_topk[qi]), answers)
    for k in K_VALUES:
        if 0 <= fr < k: hit["dpr_full"][k] += 1
    # (C) DPR filtered by exact person-name substring
    cand = []
    seen = set()
    for name in gl[qi]["persons"]:
        nl = name.lower()
        try:
            hits = searcher.search(f'"{name}"', k=KCAND)
        except Exception:
            hits = []
        for h in hits:
            if h.docid in seen: continue
            if nl in contents_of(h.docid).lower():     # exact substring verify
                seen.add(h.docid); cand.append(h.docid)
    cand_sizes.append(len(cand))
    if cand:
        C = np.stack([emb_of(p) for p in cand])        # (nc,768)
        order = np.argsort(-(C @ q))
        ranked = [cand[o] for o in order][:100]
        fr = found_rank(ranked, answers)
        for k in K_VALUES:
            if 0 <= fr < k: hit["dpr_filt"][k] += 1
    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{len(person_idx)} ({time.time()-t0:.0f}s, avg cand {np.mean(cand_sizes):.0f})", flush=True)

N = len(person_idx)
print(f"\n=== Person subset (GLiNER-large), N={N} ===")
print(f"{'system':12} " + " ".join(f"top{k:>3}" for k in K_VALUES))
for s, label in [("bm25", "BM25"), ("dpr_full", "DPR-NQ full"), ("dpr_filt", "DPR-NQ name-filt")]:
    print(f"{label:12} " + " ".join(f"{100*hit[s][k]/N:5.1f}" for k in K_VALUES))
print(f"\navg candidates per question after name-filter: {np.mean(cand_sizes):.1f} "
      f"(median {np.median(cand_sizes):.0f}, max {np.max(cand_sizes)}), "
      f"questions with 0 candidates: {sum(1 for c in cand_sizes if c==0)}")
json.dump({"N": N, "hit": {s: dict(hit[s]) for s in hit}, "avg_cand": float(np.mean(cand_sizes))},
          open(f"{SP}/filtered_dpr_results.json", "w"), indent=2)
print(f"done in {(time.time()-t0)/60:.1f} min")
