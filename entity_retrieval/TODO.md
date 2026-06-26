# TODO / Roadmap — toward *Deep Hybrid GPU Retrieval*

**Working paper title:** *Deep Hybrid GPU Retrieval*

This is the actionable roadmap. For orientation and threats-to-validity see
[`HANDOFF.md`](HANDOFF.md); for the results so far see [`README.md`](README.md).

## Thesis (two contributions)

1. **Diagnosis + method (done in prototype).** DPR's weakness on entity questions is a *retrieval*
   failure, not a *ranking* failure. Hard-constraining candidates to passages that mention the
   query's named entity (positional inverted index) and fusing the entity-filtered dense ranking
   with the global dense nearest-neighbors via RRF beats BM25 on person questions — modestly with
   zero-shot DPR (top-20 72.7 vs 71.2), clearly with fine-tuned DPR (82.4 vs 71.2).
2. **Systems contribution (to build): a fused CUDA kernel** that makes hybrid entity-filtered
   retrieval a single GEMM. See below — this is the paper's novelty engine.

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

1. **[CRITICAL] Generic hybrid baseline: BM25 ⊕ global-DPR RRF, NO entity filter.** Proves the
   *entity constraint* adds value beyond ordinary hybridization. Cheap — reuse `bm25_results/` ranked
   lists + the global DPR NN already in `filtered_rrf.py`; add as a 6th system. Until this is run the
   headline claim is not defensible.
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
7. **Fusion ablations** — `K_NN`, `K_CAND`, RRF constant, exact-match bonus (invisible at our cutoffs
   so far — justify or drop); compare RRF vs convex combination vs learned fusion (see Bruch below).

## Related work to position against (do the lit review first)

- **Bruch et al. — analysis of hybrid (lexical–semantic) search / fusion.** Most directly relevant
  to our fusion: theoretical and empirical analysis of fusion functions (RRF vs convex combination,
  score normalization, the conditions under which hybrid helps). Our RRF choice and any learned-fusion
  ablation must be framed against this; also Bruch's *Foundations of Vector Retrieval* for the
  retrieval-systems framing that the CUDA kernel slots into.
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
