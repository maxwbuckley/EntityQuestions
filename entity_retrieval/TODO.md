# TODO / Roadmap — toward *Deep Hybrid GPU Retrieval*

**Working paper title:** *Deep Hybrid GPU Retrieval*

This is the actionable roadmap. For orientation and threats-to-validity see
[`HANDOFF.md`](HANDOFF.md); for the results so far see [`README.md`](README.md).

## Thesis (two contributions)

1. **Method: fix a semantic engine's exact-match weakness *in-engine*.** The premise is a dense
   search engine that is bad at exact/entity matches. The textbook fix is to bolt on a lexical
   engine (BM25) and fuse. We show you don't have to: detect the entity, mask the dense scan with an
   entity->passage bitmap, fuse masked + global dense. **With a fine-tuned encoder this is a
   statistical dead heat with the best BM25 hybrid (83.1 vs 83.1 top-20, p=0.89) while adding no
   second retrieval system** — dense alone is 75.0. On a zero-shot encoder it lifts 40.8 -> 72.6,
   matching a plain BM25 hybrid; there a lexical arm *does* add orthogonal signal and the 3-way
   fusion wins (76.6, p<0.001). **Unifying finding: the entity mask and BM25 are redundant when the
   dense encoder is good and complementary when it isn't** — because BM25 is itself an entity filter
   (statistically, and computationally via WAND's rarest-term pruning, which usually prunes on the
   entity token). Scope limit, stated honestly: *if you already run BM25*, the mask adds ~nothing on
   a strong encoder (§7).
2. **Systems contribution (to build): a fused CUDA kernel** that makes the masked dense scan a single
   GEMM. This is no longer a bolt-on — it *is* the method. See below.

## ⭐ Systems contribution: the fused hybrid top-k CUDA kernel

The Python prototype (`scripts/filtered_rrf.py`) does the hybrid inefficiently: a separate global
nearest-neighbor pass, plus a per-question gather of name-filtered candidate embeddings, plus
host-side RRF. We even hit the obvious failure mode — materializing the full `(n_queries × n_passages)`
score matrix OOMs the GPU (21 GB for 10.7k queries), forcing query tiling.

**The kernel we want:** input is a list of `(query_vector, bitset)` pairs, where the bitset marks
the passages that mention the query's entity (the inverted-index postings, materialized as a bitset
over the corpus). In **a single fused GEMM** over the passage matrix, for each query row compute and
return **both**:
- the **top-k global** (over all columns), and
- the **top-k filtered** (over only the bitset-masked columns),

without ever materializing the full score matrix — fuse the top-k reduction into the GEMM's
tiling (à la fused-softmax/FlashAttention-style online top-k per output tile, masking columns by the
bitset for the filtered stream). One pass over the passage shards yields both ranked lists per query;
RRF (or a learned fusion) is then a trivial host-side merge of two short lists.

Why it matters:
- Turns "two retrievals + a gather" into one memory-bounded kernel — the headline GPU-systems result.
- The bitset filter is free relative to the GEMM (it's a per-tile mask), so entity-constrained
  retrieval costs ~the same as a plain dense scan.
- Generalizes: the bitset can encode *any* hard constraint (entity, metadata, ACL, freshness), so
  "filtered dense retrieval as a masked GEMM" is the reusable primitive.

Build/eval notes: prototype in CUDA/CUTLASS or Triton; compare against (a) FAISS IVF/flat + filter,
(b) the two-pass Python baseline here, (c) pre-filtering vs post-filtering. Report throughput,
recall@k (exact vs approximate), and the crossover where masking beats brute force.

## Experiments (prioritized)

1. **[DONE — see README §7] The full fusion panel, both encoders, held-out alpha, paired bootstrap.**
   `scripts/hybrid_fusion.py` runs 13 systems on one shared encoder per panel; `bootstrap_sig.py` does
   the significance tests off the per-question first-hit dump (no GPU). Settled:
   - Entity-mask fusion **ties** the best BM25 hybrid on the fine-tuned encoder (83.1 vs 83.1, p=0.89)
     with **no lexical engine**; 3-way adds nothing there (p=0.38).
   - On the zero-shot encoder the **3-way fusion is the best system** (76.6; +1.63 over the next best,
     p<0.001) — the entity mask and BM25 decorrelate when the dense arm is weak.
   - **Exact-name boosting the BM25 arm** is a free +3.0 top-20 (p<0.001) and, inside a hybrid, +1.0.
   - **CC vs RRF:** untuned (alpha=0.5) RRF wins on both encoders; tuned CC only wins when the arms are
     quality-imbalanced. **Bruch's default alpha=0.8 collapses to 46.7 top-20 on the weak encoder.**
   - Remaining: an honest test of the "no lexical engine" claim requires **building the standalone
     entity->docid index** (ours is derived from a Lucene positional index) and a real latency benchmark.

2. **Generalize beyond persons** to orgs/locations/works (GLiNER already detects them; rerun
   `run_gliner_large.py` with more labels and extend the bitset filter to any named entity).
3. **More datasets / benchmarks:**
   - **PopQA** — explicitly long-tail entity QA; the ideal stress test for the entity-retrieval thesis.
   - **BEIR** — for breadth/generalization and because it's the standard hybrid-retrieval yardstick
     (lets us compare directly to published hybrid numbers).
   - NQ / TriviaQA / WebQuestions on the same `psgs_w100` corpus (cheap, reuse index + embeddings).
4. **Downstream QA** — feed top-k to a reader/LLM and measure answer accuracy, not just recall@k.
5. **Fix the entity-mention step** (12.7% zero-candidate; diacritic folding barely helped). Try
   entity *linking* + Wikidata aliases instead of exact-substring; report entity-mention recall as a
   first-class metric (it upper-bounds the filter).
6. **Statistics** — per-relation CIs, paired bootstrap significance for RRF-vs-BM25 (close at top-20
   for zero-shot DPR).
7. **Fusion ablations** — convex combination (min-max normalized) is implemented and swept in §7/§7b.
   **Finding: whether CC beats RRF depends on arm balance.** On the fine-tuned encoder (balanced arms)
   **RRF ≥ CC** (82.0 vs 81.9; 82.8 vs 82.2 top-20). On the zero-shot encoder (imbalanced — dense arm
   near-useless) **CC > RRF** (73.1 vs 72.7; 43.9 vs 37.3 top-1), because CC's α downweights the weak
   arm (best α≈0.1–0.4). Crucially **Bruch's default α=0.8 is catastrophic on NQ (46.8 top-20)** — CC's
   win is entirely contingent on tuning α to arm quality. Net: with α tuned, CC ≥ RRF, gap growing as
   arms become imbalanced — a refinement of Bruch's "CC > RRF." Remaining: tune α on a held-out split
   (we tuned on the eval subset — an oracle upper bound), ablate `K_NN`/`K_CAND`/η and the exact-match
   bonus, try a learned fusion.

## Related work to position against (do the lit review first)

- **Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"**
  (arXiv [2210.11934](https://arxiv.org/abs/2210.11934), ECIR 2023). The key reference for our
  fusion step, and it has a direct implication: they find **convex combination (CC) of normalized
  scores outperforms RRF**, that **RRF is parameter-sensitive**, and that CC is robust to score
  normalization and tunes a single parameter with little data. This is a flag on our current setup —
  we used RRF with a hand-set constant (`c=60`) and an exact-match bonus, i.e. exactly the
  parameter-sensitivity they critique. **Action: make CC the primary fusion and treat RRF as a
  baseline** (see experiment #7).
- **SPAR / Salient-Phrase-Aware Dense Retrieval** — teaches dense models lexical matching; closest in
  spirit to "give DPR lexical/entity precision."
- **EntityQuestions** (Sciavolino et al., 2021) — already diagnoses tail-entity failure; our delta
  must be sharper than confirming it.
- **GENRE** (autoregressive entity retrieval), Entities-as-Experts, entity-linking-augmented retrieval.
- **ColBERT / late interaction**, **SPLADE / learned sparse** — get lexical precision "for free";
  our pitch is that a masked-GEMM hybrid is simpler and GPU-native.
- Standard **BM25⊕DPR hybrid** retrieval (the baseline of experiment #1).

## Threats to validity (carry into any draft)

See [`HANDOFF.md`](HANDOFF.md#threats-to-validity-state-these-in-any-draft): distant-supervision
training data, the P106-only 300-sample, observed-not-benchmarked throughput, brittle exact-substring
matching, and the `K_CAND` recall/speed trade.
