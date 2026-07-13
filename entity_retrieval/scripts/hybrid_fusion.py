"""Hybrid-retrieval fusion comparison on the GLiNER person subset.

Compares, ALL on a single shared DPR encoder (so the head-to-head is valid):

  Lexical / dense singletons
    bm25         plain BM25
    bm25_exact   BM25 with exact-entity-match boosting (lexical + entity precision)
    dpr_full     global dense top-K_NN
    dpr_filt     entity name-filtered dense (positional inverted index) ranking

  Generic hybrid (Bruch et al. 2210.11934 -- "union of lexical and semantic search"),
  NO entity filter:
    hyb_rrf      RRF( BM25 , global-DPR )
    hyb_cc       convex combination  a*phi(DPR) + (1-a)*phi(BM25), min-max normalized
                 (Bruch's TM2C2 family; a=0.8 is their tuned default)

  Our entity-constrained hybrid:
    ent_rrf      RRF( name-filtered-DPR , global-DPR )
    ent_cc       convex combination of the same two dense arms

For the two convex-combination systems we sweep the mixing weight a over a grid and
report the best top-20 operating point (the full a-curve is logged). Bruch's default
a=0.8 is also reported for hyb_cc as a no-tuning reference.

Fusion functions (Bruch, Gai & Ingber, ECIR 2023):
  RRF  : f = 1/(eta + rank_lex) + 1/(eta + rank_sem)
  CC   : f = a*phi_sem(s_sem) + (1-a)*phi_lex(s_lex),  phi = min-max over the union set
"""
import os, sys, json, time
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
from collections import defaultdict
import numpy as np, torch
REPO = "${REPO}"
SP = "${WORKDIR}"
sys.path.insert(0, REPO); sys.path.insert(0, SP)
from pyserini.search.lucene import LuceneSearcher
import utils.ion as ion
from utils.has_answer_fn import string_match
from utils.tokenizers import SimpleTokenizer
from fold_util import fold

LIMIT   = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG     = sys.argv[2] if len(sys.argv) > 2 else "ft"     # nq | ft : which DPR embeddings
K_NN    = int(os.environ.get("K_NN", 1000))              # global dense neighbors
K_BM25  = int(os.environ.get("K_BM25", 1000))            # BM25 candidates
K_CAND  = int(os.environ.get("K_CAND", 1000))            # name-filter candidates cap
RRF_ETA = 60
ALPHAS  = [round(0.1*i, 1) for i in range(1, 10)]        # 0.1 .. 0.9 for the CC sweep
K_VALUES = [1, 5, 20, 100]
SHARD   = 1000448   # uniform emb-shard size (977*1024); see encode_passages.py / README
tok = SimpleTokenizer()

meta = json.load(open(f"{SP}/dpr_{TAG}_question_meta.json"))
gl   = json.load(open(f"{SP}/gliner_results_large.json"))
Q    = np.load(f"{SP}/dpr_{TAG}_question_emb.npy").astype(np.float32)
EMB_DIR = f"{SP}/dpr_{TAG}_passage_emb"
pid2title = ion.read_json(f"{SP}/pid2title.json")
s_fold = LuceneSearcher(f"{SP}/bm25_index_folded"); s_fold.set_bm25(0.9, 0.4)
s_orig = LuceneSearcher(f"{SP}/bm25_index");        s_orig.set_bm25(0.9, 0.4)
shards = [np.load(f"{EMB_DIR}/emb_{i:03d}.npy", mmap_mode="r")
          for i in range(len(os.listdir(EMB_DIR)) // 2)]

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

# ---------- fusion primitives ----------
def rrf_fuse(listA, listB, eta=RRF_ETA):
    sc = defaultdict(float)
    for r, p in enumerate(listA): sc[p] += 1.0 / (eta + r + 1)
    for r, p in enumerate(listB): sc[p] += 1.0 / (eta + r + 1)
    return sorted(sc, key=lambda p: -sc[p])

def minmax(pid_score):
    if not pid_score: return {}
    vs = list(pid_score.values()); lo = min(vs); hi = max(vs)
    if hi <= lo: return {p: 1.0 for p in pid_score}
    return {p: (v - lo) / (hi - lo) for p, v in pid_score.items()}

def cc_fuse(scoreLex, scoreSem, alpha):
    """convex combination: alpha on the SEM arm, (1-alpha) on the LEX arm; min-max norm."""
    nL = minmax(scoreLex); nS = minmax(scoreSem)
    keys = set(nL) | set(nS)
    f = {p: alpha * nS.get(p, 0.0) + (1 - alpha) * nL.get(p, 0.0) for p in keys}
    return sorted(f, key=lambda p: -f[p])

def exact_boost(bm25_pids, folded_names):
    """lexical + entity precision: exact-name passages first (in BM25 order), then the rest."""
    exact, rest = [], []
    for p in bm25_pids:
        fc = folded_contents(p).lower()
        (exact if any(fn in fc for fn in folded_names) else rest).append(p)
    return exact + rest

person_idx = [i for i, g in enumerate(gl) if g["has_person"]]
if LIMIT:
    person_idx = person_idx[:LIMIT]
print(f"person-questions: {len(person_idx)}  TAG={TAG}  K_NN={K_NN} K_BM25={K_BM25}", flush=True)

# ---------- list B: global exact top-K_NN dense NN (scores kept), query-tiled on GPU ----------
dev = "cuda"
Qsub = torch.from_numpy(Q[person_idx]).to(dev).half()
N = Qsub.shape[0]
best = torch.full((N, K_NN), float("-inf"), device=dev, dtype=torch.float16)
bidx = torch.full((N, K_NN), -1, device=dev, dtype=torch.int64)
offset = 0; all_pids = []; t0 = time.time()
QB = int(os.environ.get("QB", 1024))
for i in range(len(shards)):
    emb = torch.from_numpy(np.load(f"{EMB_DIR}/emb_{i:03d}.npy")).to(dev).half()
    pids_i = json.load(open(f"{EMB_DIR}/pids_{i:03d}.json"))
    embT = emb.t().contiguous()
    for qs in range(0, N, QB):
        sc = Qsub[qs:qs+QB] @ embT
        v, idx = sc.topk(min(K_NN, sc.shape[1]), dim=1)
        cs = torch.cat([best[qs:qs+QB], v], 1); cg = torch.cat([bidx[qs:qs+QB], idx + offset], 1)
        nv, ni = cs.topk(K_NN, dim=1)
        best[qs:qs+QB] = nv; bidx[qs:qs+QB] = torch.gather(cg, 1, ni)
    all_pids.extend(pids_i); offset += len(pids_i); del emb, embT
all_pids = np.array(all_pids)
nn_pids   = all_pids[bidx.cpu().numpy()]      # (N, K_NN)
nn_scores = best.float().cpu().numpy()        # (N, K_NN)
print(f"global NN computed in {time.time()-t0:.0f}s", flush=True)

# ---------- per-question evaluation ----------
SIMPLE  = ["bm25", "bm25_exact", "dpr_full", "dpr_filt", "hyb_rrf", "ent_rrf"]
hit = {s: defaultdict(int) for s in SIMPLE}
# CC systems: hits[system][alpha][k]
cc_hit = {s: {a: defaultdict(int) for a in ALPHAS} for s in ("hyb_cc", "ent_cc")}
# Bruch's default alpha=0.8 is already covered by the sweep grid (0.8 in ALPHAS).
zero_cand = 0; cand_sizes = []; t0 = time.time()

def first_hit(ranked, is_ans, kmax=100):
    for r, p in enumerate(ranked[:kmax]):
        if is_ans(str(p)): return r
    return -1

def tally(store, ranked, is_ans):
    fr = first_hit(ranked, is_ans)
    for k in K_VALUES:
        if 0 <= fr < k: store[k] += 1

for n, qi in enumerate(person_idx):
    answers = meta[qi]["answers"]; q = Q[qi]; question = meta[qi]["question"]
    _isc = {}
    def is_ans(pid, _a=answers):
        if pid not in _isc:
            _isc[pid] = string_match({"text": body_of(pid)}, _a, tok)
        return _isc[pid]

    # BM25 (lexical) list + scores
    try: bm = s_orig.search(question, k=K_BM25)
    except Exception: bm = []
    bm_pids = [h.docid for h in bm]; bm_score = {h.docid: h.score for h in bm}
    tally(hit["bm25"], bm_pids, is_ans)
    # exact-match-boosted BM25
    folded_names = [fold(nm).lower() for nm in gl[qi]["persons"]]
    tally(hit["bm25_exact"], exact_boost(bm_pids, folded_names), is_ans)

    # global dense list + scores
    g_pids = [str(p) for p in nn_pids[n]]
    g_score = {str(p): float(s) for p, s in zip(nn_pids[n], nn_scores[n])}
    tally(hit["dpr_full"], g_pids, is_ans)

    # entity name-filtered dense list + scores
    cand, seen = [], set()
    for nm in gl[qi]["persons"]:
        fn = fold(nm).lower()
        try: hits = s_fold.search(f'"{fold(nm)}"', k=K_CAND)
        except Exception: hits = []
        for h in hits:
            if h.docid in seen: continue
            if fn in folded_contents(h.docid).lower():
                seen.add(h.docid); cand.append(h.docid)
    cand_sizes.append(len(cand))
    if not cand: zero_cand += 1
    f_pids, f_score = [], {}
    if cand:
        C = np.stack([emb_of(p) for p in cand]); sims = C @ q
        order = np.argsort(-sims)
        f_pids = [cand[o] for o in order]
        f_score = {cand[o]: float(sims[o]) for o in order}
        tally(hit["dpr_filt"], f_pids, is_ans)

    # generic hybrid (Bruch union): BM25 (+) global dense
    tally(hit["hyb_rrf"], rrf_fuse(bm_pids, g_pids), is_ans)
    for a in ALPHAS:
        tally(cc_hit["hyb_cc"][a], cc_fuse(bm_score, g_score, a), is_ans)

    # our entity-constrained hybrid: name-filtered dense (+) global dense
    tally(hit["ent_rrf"], rrf_fuse(f_pids, g_pids), is_ans)
    for a in ALPHAS:
        tally(cc_hit["ent_cc"][a], cc_fuse(f_score, g_score, a), is_ans)

    if (n+1) % 500 == 0:
        print(f"  {n+1}/{len(person_idx)} ({time.time()-t0:.0f}s)", flush=True)

M = len(person_idx)
def row(store): return {k: 100*store[k]/M for k in K_VALUES}

# best-alpha operating point for each CC system (by top-20)
def best_alpha(system, grid):
    curve = {a: row(cc_hit[system][a]) for a in grid}
    a_star = max(curve, key=lambda a: curve[a][20])
    return a_star, curve

hyb_cc_astar, hyb_cc_curve = best_alpha("hyb_cc", ALPHAS)
ent_cc_astar, ent_cc_curve = best_alpha("ent_cc", ALPHAS)

results = {s: row(hit[s]) for s in SIMPLE}
results["hyb_cc"]      = {"alpha_star": hyb_cc_astar, **hyb_cc_curve[hyb_cc_astar]}
results["hyb_cc_a0.8"] = {"alpha": 0.8, **row(cc_hit["hyb_cc"][0.8])}
results["ent_cc"]      = {"alpha_star": ent_cc_astar, **ent_cc_curve[ent_cc_astar]}

print(f"\n=== Hybrid fusion, person subset N={M}, DPR={TAG.upper()} (shared encoder) ===")
print(f"{'system':22} " + " ".join(f"top{k:>3}" for k in K_VALUES))
order = [("bm25","BM25"),("bm25_exact","BM25 + exact-boost"),("dpr_full",f"DPR-{TAG} global"),
         ("dpr_filt",f"DPR-{TAG} name-filt"),("hyb_rrf","Hybrid RRF (Bruch)"),
         ("hyb_cc_a0.8","Hybrid CC  a=0.8"),("hyb_cc",f"Hybrid CC  a*={hyb_cc_astar}"),
         ("ent_rrf","Entity RRF (ours)"),("ent_cc",f"Entity CC  a*={ent_cc_astar}")]
for key, label in order:
    r = results[key]
    print(f"{label:22} " + " ".join(f"{r[k]:5.1f}" for k in K_VALUES))
print(f"\nzero-candidate (name filter): {zero_cand}/{M} ({100*zero_cand/M:.1f}%), "
      f"avg cand {np.mean(cand_sizes):.1f}")
print("hyb_cc alpha curve (top20): " + " ".join(f"{a}:{hyb_cc_curve[a][20]:.1f}" for a in ALPHAS))
print("ent_cc alpha curve (top20): " + " ".join(f"{a}:{ent_cc_curve[a][20]:.1f}" for a in ALPHAS))
print(f"done in {(time.time()-t0)/60:.1f} min")

out = {"tag": TAG, "N": M, "zero_cand": zero_cand, "avg_cand": float(np.mean(cand_sizes)),
       "eta": RRF_ETA, "alpha_grid": ALPHAS,
       "hyb_cc_curve": {str(a): hyb_cc_curve[a] for a in ALPHAS},
       "ent_cc_curve": {str(a): ent_cc_curve[a] for a in ALPHAS},
       "systems": results}
json.dump(out, open(f"{SP}/hybrid_{TAG}_results.json", "w"), indent=2)
print(f"saved {SP}/hybrid_{TAG}_results.json")
