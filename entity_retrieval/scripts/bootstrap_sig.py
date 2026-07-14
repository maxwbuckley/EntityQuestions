"""Paired bootstrap over questions for the key head-to-heads (uses the per-question
first-hit dumps, so no retrieval re-run is needed)."""
import numpy as np, sys, json
K = 20; B = 10000
rng = np.random.default_rng(0)
PAIRS = [
    ("hybx_rrf", "ent_rrf",  "BM25+exact hybrid  vs  entity-masked (ours)"),
    ("hybx_rrf", "hyb_rrf",  "BM25+exact hybrid  vs  plain BM25 hybrid"),
    ("ent_rrf",  "hyb_rrf",  "entity-masked (ours) vs plain BM25 hybrid"),
    ("ent_rrf",  "dpr_full", "entity-masked (ours) vs dense alone"),
    ("tri_rrf",  "hybx_rrf", "3-way fusion       vs  BM25+exact hybrid"),
    ("tri_rrf",  "ent_rrf",  "3-way fusion       vs  entity-masked (ours)"),
    ("bm25_exact","bm25",    "BM25+exact-boost   vs  plain BM25"),
]
for tag in ("ft", "nq"):
    z = np.load(f"/home/maxwb/entityq_work/hybrid_{tag}_firsthit.npz")
    ix = z["eval_ix"]
    print(f"\n=== {tag.upper()} encoder, paired bootstrap on the EVAL split "
          f"(n={len(ix)}, B={B}, metric top-{K}) ===")
    def hits(s): 
        a = z[s][ix]; return ((a >= 0) & (a < K)).astype(np.float64)
    for A, Bs, label in PAIRS:
        ha, hb = hits(A), hits(Bs)
        d = ha - hb
        obs = 100 * d.mean()
        idx = rng.integers(0, len(d), size=(B, len(d)))
        boot = 100 * d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())      # two-sided
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns "
        print(f"  {label:44} {100*ha.mean():5.1f} vs {100*hb.mean():5.1f}  "
              f"delta {obs:+5.2f}  95% CI [{lo:+.2f},{hi:+.2f}]  p={p:.3f} {star}")
