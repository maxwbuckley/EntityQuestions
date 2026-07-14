# Handoff: where this could go, and the landmines

This note is for a researcher picking up the [`entity_retrieval/`](README.md) thread cold. It is
deliberately honest about what is **not** done. Read the [README](README.md) first for the results;
this is the "what next and what to watch out for" map. The actionable roadmap (target paper
*Deep Hybrid GPU Retrieval*, the fused CUDA kernel, the dataset list) lives in [`TODO.md`](TODO.md).

## One-paragraph summary of the finding

On entity questions, dense retrieval (DPR) underperforms BM25 mainly because it **fails to retrieve
the right entity's passages out of 21M**, not because it ranks badly. If you hard-constrain the
candidate set to passages that *mention the named entity* (via the positional inverted index) and
then rank with DPR, dense ranking is suddenly good. Fusing that entity-constrained list with the
global dense nearest-neighbors via RRF beats BM25 on person questions (top-20 72.8 zero-shot, 82.8
fine-tuned, vs 71.2). **However** (§7/§7b) a plain BM25⊕DPR hybrid with *no* entity filter already
matches that on both encoders — because BM25 supplies the same entity precision — so the *method* is
largely subsumed by standard hybrid retrieval. The cleanest, most defensible contribution is therefore
the **diagnosis** (retrieval-failure vs ranking-failure), not the entity-constrained retriever.

## Do this FIRST (before running anything else)

**A literature review to locate the novelty delta.** The method — lexical/entity filter + dense
rank + hybrid fusion — is intuitive and sits next to a lot of prior work. If you skip this you risk
building something already published. Specifically check and position against:
- **Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"**
  (arXiv [2210.11934](https://arxiv.org/abs/2210.11934)) — finds convex combination of normalized
  scores beats RRF and that RRF is parameter-sensitive; implies we should make CC our primary fusion
  and demote RRF to a baseline. See [`TODO.md`](TODO.md).
- The EntityQuestions paper itself (Sciavolino et al., 2021) — it already diagnoses the tail-entity
  failure; your delta must be sharper than "we confirmed it."
- **SPAR / "Salient Phrase Aware Dense Retrieval"** (Chen et al.) — teaches dense retrievers lexical
  matching; very close in spirit.
- BM25⊕DPR **hybrid retrieval** baselines (this is the standard, and your #1 missing baseline below).
- Autoregressive entity retrieval (**GENRE**), entity-linking-augmented retrieval, **Entities as
  Experts**, ColBERT/late-interaction, and learned sparse (**SPLADE**) which already get lexical
  precision "for free."

If the novelty survives, the framing is likely **"a diagnosis + a cheap NER-gated hybrid that
recovers it,"** not "a new retriever."

## Prioritized experiment checklist

1. **[DONE on BOTH encoders — see README §7/§7b] Generic hybrid baseline: BM25 ⊕ global-DPR, NO entity
   filter.** `scripts/hybrid_fusion.py`, full person subset. **The plain hybrid already matches the
   entity-constrained method on both encoders: 82.0 vs 82.8 top-20 (fine-tuned), 72.7 vs 72.8 (zero-shot
   NQ).** We expected the filter to help more on the weak encoder; it did not (+0.1 top-20 on NQ). The
   reason is that **BM25 is a better entity-precision signal than name-filtering dense vectors** (BM25
   71.2 vs 66.2/68.9 top-20), so the method is redundant with a standard hybrid's lexical arm. Only
   residual edge: +1.9 top-1 on the fine-tuned encoder. **New highest-value experiment:** a *three-way*
   fusion (BM25 ⊕ global-dense ⊕ entity-filtered-dense) to see if the filter adds anything orthogonal to
   BM25; if not, the method is a clean negative result and the paper is "diagnosis + hybrid suffices."
2. **Generalize beyond persons.** We only filtered on people. GLiNER detects orgs/locations/works
   too; rerun `run_gliner_large.py` with labels `["person","organization","location"]` and extend
   the filter+fusion to all named entities. A method that only works for people is a workshop paper;
   one that works for any entity is a conference paper.
3. **A second dataset / corpus.** Everything is EntityQuestions + `psgs_w100`. Add at least one of
   NQ / TriviaQA / WebQuestions (same corpus, different queries) to show it isn't EntityQuestions-
   specific. Re-using the existing index + embeddings makes this cheap.
4. **Downstream QA, not just recall.** Feed top-k passages to a reader / LLM and measure answer
   accuracy. Reviewers increasingly require this; better recall@k that doesn't move answers is weak.
5. **Fix the entity-mention step (the 12.7% zero-candidate tail).** Folding diacritics barely helped
   (12.9%→12.7%); the residual is GLiNER false-positives (e.g. "Positive K", "Staedtler") and
   non-diacritic name-form mismatches (regnal names, partial names, aliases). Try: entity *linking*
   instead of string match, alias tables (Wikidata), or relaxed token-subset matching. Report the
   entity-mention recall as a first-class metric — it upper-bounds the name-filter.
6. **Statistical rigor.** Single runs only. Add per-relation breakdowns with CIs and a significance
   test (e.g. paired bootstrap over questions) for RRF-vs-BM25, which is close at top-20 for NQ.
7. **Ablate RRF.** Vary `K_NN`, `K_CAND`, the RRF constant (c=60), and the exact-match bonus; the
   bonus was invisible at our cutoffs so either justify it or drop it. Compare RRF to learned fusion.

## Threats to validity (state these in any draft)

- **Distant-supervision training data** for fine-tuned DPR is our construction (BM25 positives +
  hard negatives), not the paper's exact Table 2 setup. The 76.2% reproduces the *finding*, not a
  bit-exact number.
- **The 300-sample is all P106** (occupation) and inflates the name-filter; always cite the
  full-subset numbers (README §6b).
- **Throughput numbers are observed, not benchmarked** (no warmup/replication); don't put them in a
  systems claim without re-measuring properly.
- **Exact-substring entity matching is brittle** and the candidate cap (`K_CAND`) trades recall
  ceiling for speed — quantify the impact (we argued it's negligible because the subject's passages
  rank top of the name query, but verify).

## How to reproduce / extend (quickstart)

> **Artifact persistence (learn from our mistake):** put `WORKDIR` on a **durable** disk, never in an
> ephemeral/temp scratch dir. The first build wrote the corpus, both indexes, the 21M×2 embeddings,
> and the fine-tuned checkpoint into a session scratchpad that got reaped between sessions — forcing a
> full multi-hour regen. Only the git-tracked scripts/docs/metric-JSONs survived. The regen is
> automated end-to-end by `scripts/master_driver.sh` (idempotent phase guards; the 2.2 h passage
> encode is resumable by shard). Note the dataset itself is **not** in this repo — download the query
> JSONs from `nlp.cs.princeton.edu/projects/entity-questions/dataset.zip` into `WORKDIR/dataset/`.

Set `WORKDIR` (≥100 GB free, durable) and `REPO` (this clone), `JAVA_HOME` (JDK 11+/21). Then, in order:
`scripts/run_gliner_large.py` → build corpus + BM25 index (`bm25/build_bm25_ctx_passages.py` +
`pyserini.index.lucene`) → `scripts/encode_passages.py`/`encode_questions.py` →
`scripts/dense_search.py`/`dense_eval.py` → `scripts/build_train_data.py`/`train_dpr.py` →
`scripts/fold_shards.py` (folded index) → `scripts/filtered_rrf.py 0 {nq,ft}`. The `*_driver.sh`
wrappers chain the long stages. Watch memory: `filtered_rrf.py` tiles the global-NN matmul (`QB`)
and caps candidates (`K_CAND`) — do **not** run two full instances in parallel (it will exhaust GPU
memory; we learned this the hard way).

## Lowest-effort, highest-value next step

Experiment #1 (generic hybrid baseline) is now **done on both encoders**, and it did *not* hold — a
plain hybrid matches the entity-constrained method, so the method is subsumed by standard hybrid
retrieval (§7/§7b). The next cheap, decisive step is the **three-way fusion** (BM25 ⊕ global-dense ⊕
entity-filtered-dense): it reuses everything in `hybrid_fusion.py` (add a third arm) and settles
whether the entity filter contributes anything *orthogonal* to BM25. If it doesn't, pivot the paper to
"diagnosis + standard hybrid suffices," and lead with the CC-vs-RRF-by-arm-balance finding (§7b) and
the GPU systems angle (`TODO.md`), not the entity-constrained retriever.
