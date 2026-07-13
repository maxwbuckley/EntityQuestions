# TODO / Roadmap — toward *Deep Hybrid GPU Retrieval*

**Working paper title:** *Deep Hybrid GPU Retrieval*

This is the actionable roadmap. For orientation and threats-to-validity see
[`HANDOFF.md`](HANDOFF.md); for the results so far see [`README.md`](README.md).

## Thesis (two contributions)

1. **Diagnosis (solid) + method (needs care after §7).** DPR's weakness on entity questions is a
   *retrieval* failure, not a *ranking* failure (§5). Hard-constraining candidates to passages that
   mention the query's named entity and fusing with global dense NN beats BM25 on person questions
   (fine-tuned DPR 82.8 vs 71.2 top-20). **But §7's critical baseline shows a plain BM25⊕DPR hybrid
   with no entity filter already reaches 82.0** — so on the fine-tuned encoder the entity constraint
   adds only a low-k precision edge (+1.9 top-1), not the bulk of the gain. The live question (now
   the paper's crux) is whether the entity constraint pays off *more* on a weak/zero-shot encoder;
   see experiment #1's NEW CRUX below.
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

1. **[DONE — and it reshapes the paper] Generic hybrid baseline: BM25 ⊕ global-DPR, no entity filter.**
   Run in §7 (`scripts/hybrid_fusion.py`, fine-tuned encoder, full person subset). **Result: the plain
   hybrid already gets 82.0 top-20 (BM25 71.2 → 82.0); the entity-constrained fusion gets 82.8 — only
   +0.8.** So on the fine-tuned encoder the entity constraint is *not* the main driver — hybridization
   is — and the filter's real value is a low-k precision edge (+1.9 top-1, +1.8 top-5). This is exactly
   the "saved you months" outcome flagged here: the headline is now the diagnosis + a precision edge,
   not "entity filtering beats BM25."
   - **[NEW CRUX] Rerun §7's table on the ZERO-SHOT NQ encoder.** The entity filter should help far more
     when the dense arm is weak on entities (NQ global DPR was 41.1 top-20 vs ft's 74.7). Hypothesis:
     *the entity constraint's marginal value is inversely related to the encoder's entity competence* —
     large when you can't fine-tune, small once you can. This is now the paper's central claim to nail.
     One command once the NQ passages are (re-)encoded: `hybrid_fusion.py 0 nq`.
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
7. **Fusion ablations** — convex combination (min-max normalized) is now implemented and swept in §7.
   **Finding so far: on this data RRF ≥ CC** (generic 82.0 vs 81.9, entity 82.8 vs 82.2 top-20), and
   CC's best α≈0.5 — *not* Bruch's 0.8 default (which gives ~78 here). So Bruch's "CC > RRF" does not
   transfer to this entity-QA + fine-tuned-dense setting; report this as a data point rather than
   adopting CC as primary. Remaining: tune α on a held-out split (we tuned on the eval subset — an
   oracle upper bound), ablate `K_NN`/`K_CAND`/η and the exact-match bonus, try a learned fusion, and
   check whether CC overtakes RRF on the weak NQ encoder (experiment #1's NEW CRUX).

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
