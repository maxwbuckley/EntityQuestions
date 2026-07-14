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

  Generic hybrid with the ENTITY-AWARE lexical arm (exact-boosted BM25 instead of BM25):
    hybx_rrf     RRF( exact-boosted-BM25 , global-DPR )
    hybx_cc      convex combination with the boosted lexical score

  Our entity-constrained hybrid:
    ent_rrf      RRF( name-filtered-DPR , global-DPR )
    ent_cc       convex combination of the same two dense arms

  Three-way fusion (does the name filter add anything ORTHOGONAL to BM25?):
    tri_rrf      RRF( BM25 , global-DPR , name-filtered-DPR )
    trix_rrf     RRF( exact-boosted-BM25 , global-DPR , name-filtered-DPR )

PROTOCOL. The person subset is split into a TUNE (30%) and a disjoint EVAL (70%) half.
The CC mixing weight a is CHOSEN on TUNE and REPORTED on EVAL; every system in the main
table is scored on the same EVAL questions. This matters: RRF has no free parameter, so
picking a by its best score on the evaluation set (as an earlier version of this script
did) is an oracle and silently overstates CC. Both numbers are printed so the size of
that inflation is visible. Bruch's default a=0.8 is also reported as a no-tuning
reference. All per-question first-hit ranks are dumped to .npz so any other split or
statistic can be computed later without re-running.

Fusion functions (Bruch, Gai & Ingber, ECIR 2023):
  RRF  : f = sum_i 1/(eta + rank_i)
  CC   : f = a*phi_sem(s_sem) + (1-a)*phi_lex(s_lex),  phi = min-max over the union set

The boosted lexical SCORE (for CC) is the score-space image of the rank-space boost;
see boosted_lex_score(). It introduces no new hyperparameter.
"""
import os, sys, json, time
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
from collections import defaultdict
import numpy as np, torch
REPO = "/mnt/c/Users/maxwb/Development/EntityQuestions"
SP = "/home/maxwb/entityq_work"
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
def rrf_fuse(*lists, eta=RRF_ETA):
    sc = defaultdict(float)
    for lst in lists:
        for r, p in enumerate(lst): sc[p] += 1.0 / (eta + r + 1)
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

def exact_split(bm25_pids, folded_names):
    """partition a BM25 list into (mentions the entity, doesn't), preserving BM25 order."""
    exact, rest = [], []
    for p in bm25_pids:
        fc = folded_contents(p).lower()
        (exact if any(fn in fc for fn in folded_names) else rest).append(p)
    return exact, rest

def boosted_lex_score(bm25_score, exact_set):
    """score-space image of the rank-space exact boost (see module docstring).

    The entity indicator is weighted 2x, i.e. strictly above the largest possible
    min-max BM25 contribution (1.0), so every exact-name passage outscores every
    non-exact one and the induced ranking is *identical* to exact_split()'s order.
    cc_fuse() min-max normalizes this again, so the absolute scale is irrelevant.
    """
    nb = minmax(bm25_score)
    return {p: v + (2.0 if p in exact_set else 0.0) for p, v in nb.items()}

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
SIMPLE  = ["bm25", "bm25_exact", "dpr_full", "dpr_filt",
           "hyb_rrf", "hybx_rrf", "ent_rrf", "tri_rrf", "trix_rrf"]
CC_SYS  = ("hyb_cc", "hybx_cc", "ent_cc")
# We store the PER-QUESTION first-hit rank (-1 = no hit in top-100) rather than running
# totals, so that any subset (e.g. a held-out eval split) can be scored after the fact.
# This is what makes an honest, non-oracle choice of the CC mixing weight alpha possible:
# alpha is selected on a TUNE split and reported on a DISJOINT EVAL split.
fh    = {s: [] for s in SIMPLE}                            # fh[system][i]
cc_fh = {s: {a: [] for a in ALPHAS} for s in CC_SYS}       # cc_fh[system][alpha][i]
# Bruch's default alpha=0.8 is already covered by the sweep grid (0.8 in ALPHAS).
zero_cand = 0; cand_sizes = []; t0 = time.time()

def first_hit(ranked, is_ans, kmax=100):
    for r, p in enumerate(ranked[:kmax]):
        if is_ans(str(p)): return r
    return -1

def tally(store, ranked, is_ans):
    store.append(first_hit(ranked, is_ans))

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
    tally(fh["bm25"], bm_pids, is_ans)
    # exact-match-boosted BM25 (entity-aware lexical arm)
    folded_names = [fold(nm).lower() for nm in gl[qi]["persons"]]
    ex_pids, rest_pids = exact_split(bm_pids, folded_names)
    bmx_pids  = ex_pids + rest_pids
    bmx_score = boosted_lex_score(bm_score, set(ex_pids))
    tally(fh["bm25_exact"], bmx_pids, is_ans)

    # global dense list + scores
    g_pids = [str(p) for p in nn_pids[n]]
    g_score = {str(p): float(s) for p, s in zip(nn_pids[n], nn_scores[n])}
    tally(fh["dpr_full"], g_pids, is_ans)

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
        tally(fh["dpr_filt"], f_pids, is_ans)
    else:
        fh["dpr_filt"].append(-1)      # no candidates -> counted as a miss, not dropped

    # generic hybrid (Bruch union): BM25 (+) global dense
    tally(fh["hyb_rrf"], rrf_fuse(bm_pids, g_pids), is_ans)
    for a in ALPHAS:
        tally(cc_fh["hyb_cc"][a], cc_fuse(bm_score, g_score, a), is_ans)

    # generic hybrid with the entity-aware lexical arm: exact-boosted BM25 (+) global dense
    tally(fh["hybx_rrf"], rrf_fuse(bmx_pids, g_pids), is_ans)
    for a in ALPHAS:
        tally(cc_fh["hybx_cc"][a], cc_fuse(bmx_score, g_score, a), is_ans)

    # our entity-constrained hybrid: name-filtered dense (+) global dense
    tally(fh["ent_rrf"], rrf_fuse(f_pids, g_pids), is_ans)
    for a in ALPHAS:
        tally(cc_fh["ent_cc"][a], cc_fuse(f_score, g_score, a), is_ans)

    # three-way: does the name-filtered dense arm add anything orthogonal to BM25?
    tally(fh["tri_rrf"],  rrf_fuse(bm_pids,  g_pids, f_pids), is_ans)
    tally(fh["trix_rrf"], rrf_fuse(bmx_pids, g_pids, f_pids), is_ans)

    if (n+1) % 500 == 0:
        print(f"  {n+1}/{len(person_idx)} ({time.time()-t0:.0f}s)", flush=True)

M = len(person_idx)
fh    = {s: np.array(v, dtype=np.int32) for s, v in fh.items()}
cc_fh = {s: {a: np.array(v, dtype=np.int32) for a, v in d.items()} for s, d in cc_fh.items()}
for s, v in fh.items():
    assert len(v) == M, f"{s}: {len(v)} != {M}"     # every question recorded, incl. misses

# ---------- tune / eval split: alpha is CHOSEN on TUNE, REPORTED on EVAL ----------
# A single held-out split is what makes the CC-vs-RRF comparison fair: RRF has no free
# parameter, so scoring CC at its best-on-the-eval-set alpha (as we did originally) is an
# oracle and overstates CC. Bruch's own claim is that alpha is cheap to tune, so a small
# tune split suffices. Split is deterministic (fixed seed) and disjoint.
TUNE_FRAC = float(os.environ.get("TUNE_FRAC", 0.30))
rng = np.random.default_rng(0)
perm = rng.permutation(M)
n_tune = int(round(TUNE_FRAC * M))
tune_ix, eval_ix = np.sort(perm[:n_tune]), np.sort(perm[n_tune:])

def row(arr, ix):
    a = arr[ix]; n = len(ix)
    return {k: float(100 * np.count_nonzero((a >= 0) & (a < k)) / n) for k in K_VALUES}

def pick_alpha(system, ix, k=20):
    """choose alpha by top-k on the given (tune) split"""
    return max(ALPHAS, key=lambda a: row(cc_fh[system][a], ix)[k])

astar   = {s: pick_alpha(s, tune_ix) for s in CC_SYS}        # honest: tuned on TUNE only
oracle  = {s: pick_alpha(s, eval_ix) for s in CC_SYS}        # oracle: tuned on EVAL (for comparison)
curves  = {s: {a: row(cc_fh[s][a], eval_ix) for a in ALPHAS} for s in CC_SYS}

# Primary table: EVAL split only, for every system (so RRF and CC are on identical questions).
results = {s: row(fh[s], eval_ix) for s in SIMPLE}
for s in CC_SYS:
    results[s] = {"alpha_star_tuned_on_tune_split": astar[s], **curves[s][astar[s]]}
    results[s + "_ORACLE"] = {"alpha_star_tuned_on_eval": oracle[s], **curves[s][oracle[s]]}
results["hyb_cc_a0.8"] = {"alpha": 0.8, **curves["hyb_cc"][0.8]}
# Full-subset numbers (all M questions) for continuity with the earlier README tables.
full = {s: row(fh[s], np.arange(M)) for s in SIMPLE}
for s in CC_SYS:
    full[s] = {"alpha_star_tuned_on_tune_split": astar[s],
               **row(cc_fh[s][astar[s]], np.arange(M))}

print(f"\n=== Hybrid fusion, DPR={TAG.upper()} (shared encoder) ===")
print(f"person subset N={M}  |  tune={len(tune_ix)}  eval={len(eval_ix)} (alpha tuned on TUNE)")
print(f"\n{'system':30} " + " ".join(f"top{k:>3}" for k in K_VALUES) + "   [EVAL split]")
order = [("bm25","BM25"),("bm25_exact","BM25 + exact-boost"),("dpr_full",f"DPR-{TAG} global"),
         ("dpr_filt",f"DPR-{TAG} name-filt"),
         ("hyb_rrf","Hybrid RRF (Bruch)"),
         ("hyb_cc_a0.8","Hybrid CC  a=0.8 (Bruch dflt)"),
         ("hyb_cc",f"Hybrid CC  a={astar['hyb_cc']} (held-out)"),
         ("hybx_rrf","Hybrid+exact RRF"),
         ("hybx_cc",f"Hybrid+exact CC a={astar['hybx_cc']} (held-out)"),
         ("ent_rrf","Entity RRF (ours)"),
         ("ent_cc",f"Entity CC  a={astar['ent_cc']} (held-out)"),
         ("tri_rrf","3-way RRF"),("trix_rrf","3-way RRF (+exact)")]
for key, label in order:
    r = results[key]
    print(f"{label:30} " + " ".join(f"{r[k]:5.1f}" for k in K_VALUES))
print("\n-- oracle alphas (tuned ON the eval set; the flawed protocol, shown for contrast) --")
for s in CC_SYS:
    o = results[s + "_ORACLE"]; h = results[s]
    print(f"{s:10} held-out a={astar[s]} -> top20 {h[20]:5.1f}   |   "
          f"oracle a={oracle[s]} -> top20 {o[20]:5.1f}   (oracle inflation {o[20]-h[20]:+.1f})")
print(f"\nzero-candidate (name filter): {zero_cand}/{M} ({100*zero_cand/M:.1f}%), "
      f"avg cand {np.mean(cand_sizes):.1f}")
for s in CC_SYS:
    print(f"{s} alpha curve on EVAL (top20): "
          + " ".join(f"{a}:{curves[s][a][20]:.1f}" for a in ALPHAS))
print(f"done in {(time.time()-t0)/60:.1f} min")

out = {"tag": TAG, "N": M, "n_tune": int(len(tune_ix)), "n_eval": int(len(eval_ix)),
       "tune_frac": TUNE_FRAC, "split_seed": 0, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
       "zero_cand": zero_cand, "avg_cand": float(np.mean(cand_sizes)),
       "eta": RRF_ETA, "alpha_grid": ALPHAS,
       "alpha_star_heldout": astar, "alpha_star_oracle": oracle,
       **{f"{s}_curve_eval": {str(a): curves[s][a] for a in ALPHAS} for s in CC_SYS},
       "systems_eval_split": results,
       "systems_full_subset": full}
json.dump(out, open(f"{SP}/hybrid_{TAG}_results.json", "w"), indent=2)
# per-question first-hit ranks, so any future split/statistic needs no re-run (paired
# bootstrap, per-relation CIs, ...). -1 = no answer-bearing passage in the top-100.
np.savez_compressed(f"{SP}/hybrid_{TAG}_firsthit.npz",
                    person_idx=np.array(person_idx), tune_ix=tune_ix, eval_ix=eval_ix,
                    **{s: v for s, v in fh.items()},
                    **{f"{s}__a{a}": cc_fh[s][a] for s in CC_SYS for a in ALPHAS})
print(f"saved {SP}/hybrid_{TAG}_results.json + hybrid_{TAG}_firsthit.npz")
