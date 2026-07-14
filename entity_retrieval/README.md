# Fixing a Semantic Search Engine's Exact-Match Weakness — Without Adding a Lexical Engine

This directory documents an end-to-end study built on top of the
[EntityQuestions](https://arxiv.org/pdf/2109.08535.pdf) dataset (Sciavolino et al., EMNLP 2021).
It starts by **reproducing the paper's central result** — that BM25 beats dense retrieval (DPR)
on simple entity-centric questions — and then asks the question a practitioner actually faces:

> You are running a **semantic search engine** (dense embeddings, exact nearest-neighbor search). It
> is great at paraphrase and **terrible at exact matches** — ask it about a named person and it
> retrieves passages about the wrong one. **What is the cheapest thing you can add to fix that?**

The textbook answer is *"bolt on BM25"* — stand up a whole second retrieval engine and fuse the two
ranked lists (Bruch et al.). This study measures an alternative that never leaves the dense engine:
**detect the entity in the query, use an entity→passage bitmap to mask the dense scan, and fuse the
masked ranking with the global one.** Same embeddings, same GEMM, no lexical scoring path.

**The headline result (§7), with significance tests:**

> **With a good dense encoder, the entity mask completely substitutes for a lexical engine.** It takes
> the semantic engine from **75.0 → 83.1** top-20 (+8.1) and **47.7 → 59.6** top-1 (+11.9), landing in a
> **statistical dead heat with the best BM25 hybrid** (83.1 vs 83.1, *p*=0.89) — while adding no second
> retrieval system. With a *weak* (zero-shot) encoder it still does most of the work (40.8 → 72.6,
> matching a plain BM25 hybrid), but there a lexical arm adds genuinely orthogonal information and the
> best system fuses all three arms (76.6).

**The unifying finding:** the entity mask and BM25 are **redundant when the dense encoder is good, and
complementary when it isn't** — because *BM25 is itself an entity filter*, statistically (it beats
name-filtered dense outright) and computationally (its rarest query term usually **is** the entity
token, which is exactly what block-max WAND prunes on).

Everything here was run against the real DPR Wikipedia corpus (`psgs_w100`, 21,015,324 passages)
on a single RTX 5090.

---

## Headline results

**Reproduction — full test set (24 relations, 22,075 questions), macro-averaged top-20:**

| Retriever | top-20 | notes |
|---|---|---|
| BM25 | **72.0%** | exact reproduction of the paper (Table 1 / Table 5) |
| DPR-NQ (zero-shot) | **50.9%** | reproduces the paper's ~49.7% (DPR ≪ BM25, the paper's thesis) |
| DPR fine-tuned on EntityQuestions train | **76.2%** | in-domain fine-tuning overtakes BM25 (paper Table 2) |

**The main comparison (§7)** — person questions, held-out eval split (n=7,531), one shared encoder per
panel, α tuned on a disjoint split, significance by paired bootstrap. Grouped by *what infrastructure
each system requires*:

| System | ft top-20 | zero-shot top-20 | needs a lexical engine? |
|---|---|---|---|
| Dense global — *the engine you start with* | 75.0 | 40.8 | **no** |
| Entity-masked dense, alone | 68.6 | 65.7 | **no** |
| **Entity-masked ⊕ global dense (ours)** | **83.1** | **72.6** | **no** |
| BM25 | 70.8 | 70.8 | yes |
| BM25 + exact-name boost | 73.9 | 73.9 | yes |
| BM25 ⊕ dense, RRF (Bruch hybrid) | 82.2 | 72.4 | yes |
| BM25 + exact-boost ⊕ dense, RRF | **83.1** | 75.0 | yes |
| 3-way: BM25 ⊕ global ⊕ entity-masked | 82.9 | **76.6** | yes |

Three things to read off this table:

- **Ours ties the best lexical hybrid on the fine-tuned encoder** (83.1 vs 83.1, *p*=0.89) with no BM25
  anywhere in the system — and adding BM25 on top of it (3-way, 82.9) buys nothing (*p*=0.38).
- **On a weak encoder the arms decorrelate:** the 3-way fusion is the best system in the study (76.6,
  *p*<0.001 over the next best). A weak dense arm makes the entity mask and BM25 complementary.
- **BM25 + exact-name boost is the cheapest win here** (+3.0 top-20 over BM25, *p*<0.001, no dense
  compute at all): float passages containing the detected name to the top of the BM25 list. If you run
  a lexical stack and take one thing from this repo, take this.

**Do not copy Bruch's α=0.8 across encoders:** on the weak dense arm it collapses to **46.7** top-20,
versus **72.4** for parameter-free RRF. See §7.

The **300-question P106 sample** used during development appears in §5–§6; it is one easy relation and
overstates the name-filter, so quote §7's numbers, not those.

---

## How to read this directory

```
entity_retrieval/
  README.md            <- this file (the narrative: what we did and why)
  TODO.md              <- roadmap: target paper "Deep Hybrid GPU Retrieval", the fused CUDA
                          kernel, dataset plan (PopQA/BEIR/...), prioritized experiments
  HANDOFF.md           <- orientation for a researcher picking this up cold + threats to validity
  scripts/             <- every script we ran, in order of the story below
  results/             <- saved metric JSONs
```

**Picking this up to write a paper?** Start with [`HANDOFF.md`](HANDOFF.md) then [`TODO.md`](TODO.md).

The scripts use two path placeholders you must set to reproduce:
- `${WORKDIR}` / `WORKDIR` — a scratch dir with ~100 GB free (corpus, index, embeddings live here)
- `${REPO}` — a clone of this repository (for `utils/` and `relation_query_templates.json`)

Environment used: Python 3.13, PyTorch 2.11 (CUDA), `transformers`, `gliner`, `pyserini`
(Lucene/Anserini, needs `JAVA_HOME` pointing at a JDK 11+/21), `faiss` not required.

---

## 0. Background: what EntityQuestions is and why it matters

EntityQuestions is a set of simple, templated questions about (mostly tail) Wikidata entities,
e.g. *"Where was [X] born?"*, *"Who is [X] married to?"*. There are 24 relations; each question
has a short answer string. Retrieval is evaluated as **top-k recall**: does at least one of the
top-k retrieved Wikipedia passages contain the answer string? (`utils/has_answer_fn.py`).

The corpus you retrieve over is **not** in this repo — it's the standard DPR Wikipedia split
`psgs_w100` (21M 100-word passages), downloaded separately. The train/dev/test JSON files are the
*queries*, not the corpus.

The paper's punchline: **dense retrievers (DPR) badly underperform BM25** on these questions,
because tail entities are rarely seen in DPR's training data, so DPR struggles to map an
entity name to its passages.

---

## 1. How many questions even name a person? (GLiNER)

**Why:** Our entity-constraint idea only applies to questions that name a concrete entity. We
focus on *people*. First we measure how many questions name a person, using GLiNER NER.

**What:** `scripts/run_gliner.py` (medium) and `scripts/run_gliner_large.py` (large-v2.1, "XL"),
single label `person`, threshold 0.5, over all 22,075 test questions.

**Result:**

| Model | Questions naming ≥1 person | % |
|---|---|---|
| `gliner_medium-v2.1` | 10,153 / 22,075 | 46.0% |
| `gliner_large-v2.1` ("XL") | 10,757 / 22,075 | 48.7% |

The split is almost binary by relation: person-subject relations (P26 spouse, P19 birthplace,
P106 occupation, P40 child, P69 educated, P413 position) are ~99–100% person; geographic/org
relations (P36 capital, P17 country) are ~0%. The large model recovers ~30 near-threshold misses
the medium model dropped (royalty/nobility with "[Name] of [Place]" forms scored just under 0.5).
We use the **large** model's person set for everything downstream.

---

## 2. Reproducing BM25 (exact match to the paper)

**Why:** Establish a faithful baseline and validate the whole pipeline against a known number.

**What:**
1. Download `psgs_w100.tsv.gz` (4.7 GB) from the DPR repo; decompress (21,015,324 passages).
2. `bm25/build_bm25_ctx_passages.py` (in the repo root) → JSON shards + passage-id→title map.
3. Build a Lucene BM25 index with Pyserini (`-storePositions -storeDocvectors -storeRaw`),
   default BM25 (k1=0.9, b=0.4), Porter stemming, English stopwords. 21M docs index in ~4 min.
4. `scripts/bm25_retrieve.py` (the repo's `bm25/bm25_retriever.py`, adapted from the old
   `SimpleSearcher` API to the current `LuceneSearcher`) retrieves top-100 for all test questions.
5. `utils/accuracy.py` computes macro-averaged top-k recall.

**Result: macro top-20 = 72.0%, identical to the paper, all 24 relations matching Table 5 to
within 0.0 points.**

One subtlety we pinned down: the repo's recent bug-fix commit (c4969aa) excludes the passage
*title* from answer-matching, which gives **71.2%** top-20; the paper's published **72.0%** used
the older title-inclusive matching. Both are reproduced exactly depending on the
`--answer_match` mode. Corrected (title-excluded) top-1/5/20/100 macro = **43.4 / 60.8 / 71.2 / 79.8**.

---

## 3. Reproducing zero-shot DPR-NQ (the dense gap)

**Why:** Reproduce the paper's central comparison — that DPR-NQ ≪ BM25 on entity questions.

**What (and some engineering honesty):**
- We load the **canonical DPR-NQ checkpoint** (`hf_bert_base.cp` from the DPR repo). HuggingFace
  downloads were unreliable in our environment, so `scripts/dpr_common.py` loads the original
  `.cp` directly: it registers a stub for DPR's `CheckpointState` namedtuple so `torch.load` can
  unpickle it, maps the `question_model.*` / `ctx_model.*` BERT weights into HF `BertModel`s, and
  uses DPR's representation = the **`[CLS]` last-hidden-state** (not BERT's tanh pooler).
  We validated this is faithful: the loaded weights are **byte-identical** to the official HF DPR
  encoder, and our `[CLS]` output **exactly matches** `DPRQuestionEncoder.pooler_output` (max diff 0.0).
- `scripts/encode_passages.py` encodes all 21M passages (fp16, ~2,700 psg/s, ~2.2 h) into 22 shards.
- `scripts/encode_questions.py` encodes the test questions.
- `scripts/dense_search.py` does **exact** top-100 search by streaming the embedding shards through
  the GPU and keeping a running top-k. We do this instead of a FAISS flat index for a concrete
  reason: a flat index over 21M×768 fp32 needs ~65 GB RAM and the box has 47 GB. Streaming on the
  GPU does the full 21M×22K exact search in **~27 seconds** and sidesteps the RAM limit entirely.
- `scripts/dense_eval.py` computes top-k recall.

**Result: macro top-20 = 50.9%** vs the paper's 49.7% (top-1/5/20/100 = 25.0 / 39.4 / 50.9 / 64.3).
The per-relation pattern reproduces the thesis exactly — DPR collapses on entity-heavy relations:

| Relation | BM25 top-20 | DPR-NQ top-20 |
|---|---|---|
| P26 spouse | 89.7 | **41.5** |
| P19 birthplace | 75.3 | **32.0** |
| P40 child | 85.0 | **20.8** |

---

## 4. Fine-tuning DPR on EntityQuestions (closing the gap)

**Why:** The paper (Table 2) shows in-domain fine-tuning makes DPR competitive. We reproduce that,
and we'll need a strong dense encoder for the fusion experiment later.

**What:** The dataset only ships `{question, answers}` — no DPR-format positives/negatives — so we
build training data by **distant supervision** (`scripts/build_train_data.py`): BM25-retrieve each
train question, take the top answer-bearing passage as the positive and top non-answer passages as
hard negatives. This yields **69,400** training examples (3,000/relation cap).
`scripts/train_dpr.py` fine-tunes a DPR bi-encoder (init from DPR-NQ, in-batch + hard-negative NLL
loss, `[CLS]` reps, 10 epochs, ~37 min). Then we re-encode all 21M passages and re-retrieve.

**Result: macro top-20 = 76.2%** (up from 50.9%, and **above BM25's 72.0%**). The gains are exactly
on the entity-heavy relations where zero-shot DPR collapsed:

| Relation | DPR-NQ zero-shot | DPR fine-tuned | BM25 |
|---|---|---|---|
| P19 born | 32.0 | **70.9** | 75.3 |
| P26 spouse | 41.5 | **72.4** | 89.7 |
| P40 child | 20.8 | **66.4** | 85.0 |
| P413 position | 77.5 | **94.4** | 74.3 |
| **Macro top-20** | **50.9** | **76.2** | 72.0 |

---

## 5. The core idea: entity-constrained dense retrieval

**Why:** Is DPR's entity weakness about *ranking* or about *finding the right entity among 21M*?
Test: constrain the candidate set to passages that mention the person, then rank with DPR.

**What (`scripts/filtered_dpr.py`):** for each person question, use the **positional inverted
index** to find passages whose contents contain the person name (a phrase/"bigram" query — exactly
why we stored positions at index time), verify the exact name substring, then rank *only those
passages* by DPR vector similarity. Compare BM25 vs DPR-full vs DPR-name-filtered on the person
subset.

**Result (full 10,757-question person subset, DPR-NQ):**

| System | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|
| BM25 | 41.2 | 59.8 | 71.2 | 80.2 |
| DPR-NQ full | 15.1 | 28.2 | 41.1 | 58.0 |
| DPR-NQ name-filtered | 35.8 | 57.4 | 65.7 | 68.6 |

**The hard filter lifts DPR's top-20 from 41.1 → 65.7 (+24.6).** This is the key evidence: once you
constrain to the right entity, DPR ranks well — its weakness was *retrieval of the entity*, not
ranking. But name-filtering alone doesn't beat BM25, for two reasons: (1) ~12.9% of questions get
**zero candidates** (the name isn't found — diacritics, name variants, GLiNER false positives),
which are pure 0-recall losses; (2) the hard filter caps top-100 recall.

---

## 6. Fixing the limitations: RRF fusion + diacritic folding + tie-break

**Why:** Address both failure modes. (a) Give zero-candidate questions a fallback. (b) Reduce the
zero-candidate rate at its source.

**What:**
- **Diacritic folding before indexing & search** (`scripts/fold_util.py`, `scripts/fold_shards.py`):
  rebuild the inverted index with diacritics folded (`Mihály`→`Mihaly`, plus an explicit map for
  non-decomposing letters `Ł→L`, `Ø→O`, `æ→ae`, `ß→ss`, …), and fold the query name identically.
  This makes the name match robust to accent mismatches between GLiNER and the corpus.
- **Reciprocal Rank Fusion** (`scripts/filtered_rrf.py`): grab two ranked lists per question —
  (A) the name-filtered DPR ranking (from the postings index) and (B) the **global exact dense
  nearest-neighbors** (top-1000) — and fuse with RRF (`score = Σ 1/(60 + rank)`). List B is the
  fallback that covers zero-candidate questions; list A supplies entity precision.
- **Exact-match tie-break:** a tiny additive weight on list-A (exact-match) members so ties break
  toward the exact match.

**Result (300-question sample, P106), DPR-NQ embeddings:**

| System | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|
| BM25 | 41.7 | 58.0 | 69.7 | 85.0 |
| DPR-NQ full | 5.3 | 16.0 | 29.7 | 60.7 |
| DPR-NQ name-filtered (folded) | 36.3 | 65.3 | 74.7 | 78.7 |
| **RRF fuse** | 36.3 | 62.0 | **78.3** | **90.0** |
| RRF + exact-bonus | 36.3 | 62.3 | 78.3 | 90.0 |

RRF fixes the recall ceiling (name-filter's 78.7 → **90.0** at top-100, via the global-NN fallback)
while keeping low-k precision. With **fine-tuned** DPR embeddings it's even stronger:

| System (300-sample, DPR-ft) | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|
| BM25 | 41.7 | 58.0 | 69.7 | 85.0 |
| DPR-ft full | 55.3 | 74.7 | 88.0 | 94.0 |
| DPR-ft name-filtered | 66.3 | 74.7 | 77.7 | 78.7 |
| **DPR-ft + RRF** | **69.7** | **82.0** | **92.3** | **95.7** |

**Fine-tuned DPR + inverted-index RRF strictly dominates every individual method at every cutoff**,
and beats BM25 at top-20 by +22.6. Notably the name filter adds precision even to the strong
fine-tuned model (top-1: full 55.3 → filtered 66.3 → fused 69.7) — the exact-entity constraint
removes near-miss distractors the dense model would otherwise rank highly.

### 6b. The full picture (full person subset)

> Full 10,757-question results for both DPR-NQ and DPR-ft are produced by
> `scripts/filtered_rrf.py 0 nq` and `... 0 ft` and saved to `results/rrf_*_full_results.json`.
_The 300-sample above is one easy relation (P106) and overstates the name-filter; the full-subset
numbers below (all 10,757 person questions) are the ones to trust._

**Full person subset (N=10,757), DPR-NQ embeddings:**

| System | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|
| BM25 | 41.2 | 59.8 | 71.2 | 80.2 |
| DPR-NQ full | 15.1 | 28.2 | 41.1 | 58.0 |
| DPR-NQ name-filtered | 35.3 | 57.4 | 65.8 | 68.7 |
| **DPR-NQ + RRF** | 37.7 | 60.4 | **72.7** | **81.9** |

**Full person subset (N=10,757), fine-tuned DPR embeddings:**

| System | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|
| BM25 | 41.2 | 59.8 | 71.2 | 80.2 |
| DPR-ft full | 47.9 | 63.8 | 75.0 | 84.3 |
| DPR-ft name-filtered | 55.0 | 64.9 | 67.8 | 69.0 |
| **DPR-ft + RRF** | **60.0** | **74.5** | **82.4** | **88.5** |

**Takeaways that survive the full subset:**
- **RRF beats BM25 even with zero-shot DPR** (72.7 vs 71.2 top-20; 81.9 vs 80.2 top-100) — the
  global-NN arm rescues the ~12.7% zero-candidate questions that sink the name-filter alone (65.8).
- **With fine-tuned DPR, RRF dominates everything** — 82.4 top-20 (+11.2 over BM25), best at every k.
  _(§7 shows a plain BM25⊕DPR hybrid reaches the same place, so this is **not** evidence that the
  entity constraint beats hybridization. The right reading of §7 is that the entity mask is an
  **alternative** to a lexical engine — it buys the whole hybrid gain without one — not that it beats
  BM25 at BM25's own game.)_
- **The name filter contributes precision the dense model lacks:** at top-1 it beats full dense
  retrieval for both encoders (NQ 15.1→35.3; ft 47.9→55.0), and the fusion improves on both
  (ft top-1: full 47.9 → filtered 55.0 → fused 60.0).
- Numbers are lower than the P106-only 300-sample (e.g. ft top-20 92.3 → 82.4), as expected — but
  the ordering (RRF > full > name-filter, RRF > BM25) is unchanged.
- Diacritic folding moved the zero-candidate rate only 12.9%→12.7% on the full set: the residual is
  dominated by GLiNER false-positive "names" and non-diacritic name-form mismatches, not accents.
  See `HANDOFF.md` for why this matters and what to do about it.

---

## 7. The question this study is actually answering

**The premise.** You are building a **semantic search engine** — dense embeddings, exact
nearest-neighbor search over the whole corpus. It is excellent at paraphrase and terrible at exact
matches: ask it a question about a specific named entity and it retrieves passages about the wrong
person entirely (§5: DPR-nq gets **40.8** top-20 where BM25 gets 70.8). **What is the cheapest thing
you can add to fix that?**

There are two families of answer, and they cost very different things to operate:

- **Add a lexical engine** (the standard hybrid; Bruch et al.) — stand up BM25/Lucene next to your
  vector index, run both, fuse the two ranked lists. A second retrieval system to build, host, tune
  and keep in sync.
- **Stay inside the dense engine** (this work) — detect the entity in the query, use an
  entity→passage bitmap to *mask* the dense scan, and fuse the masked ranking with the global one.
  Same embeddings, same GEMM, no lexical scoring path.

§7 measures both, on one shared encoder, with significance tests. The answer depends on how good
your dense encoder already is — and that dependence is the main finding.

### The reference: Bruch's fusion functions

**Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"**
(arXiv [2210.11934](https://arxiv.org/abs/2210.11934), ACM TOIS 2023) fuse a lexical ranking (BM25)
and a semantic ranking (dense) two ways:

- **Reciprocal Rank Fusion (RRF):** `f = Σᵢ 1/(η + πᵢ)` — rank-based, parameter-free in practice (η=60).
- **Convex Combination (CC):** `f = α·φ_Sem(s_Sem) + (1−α)·φ_Lex(s_Lex)`, with φ per-query
  **min-max normalization** `φ(s)=(s−m_q)/(M_q−m_q)` (their TM2C2 uses theoretical bounds: BM25 inf=0,
  cosine inf=−1). They find **CC > RRF in- and out-of-domain**, RRF is η-sensitive, CC is
  normalization-agnostic and sample-efficient, and α≈0.8 is a good default.

We implement both, plus two entity-aware systems of our own, all in
[`scripts/hybrid_fusion.py`](scripts/hybrid_fusion.py).

### The systems

| System | Lexical arm | Dense arm(s) | Needs a lexical engine? |
|---|---|---|---|
| `bm25` | BM25 | — | yes |
| `bm25_exact` | BM25, **exact-name matches floated to the top** | — | yes |
| `dpr_full` | — | global dense | **no** |
| `dpr_filt` | — | entity-masked dense | **no** |
| `hyb_*` (Bruch) | BM25 | global dense | yes |
| `hybx_*` | BM25 + exact-boost | global dense | yes |
| **`ent_*` (ours)** | — | **entity-masked ⊕ global dense** | **no** |
| `tri_rrf` | BM25 | global ⊕ entity-masked dense | yes |

The **exact-name boost** is a three-line lexical trick: partition the BM25 top-k into passages that
literally contain the detected entity name and those that don't, and put the former first (BM25 order
preserved within each group). For CC it has a score-space form that induces exactly the same ranking
(see `boosted_lex_score()`), so it adds no hyperparameter.

**Protocol.** One shared encoder per panel (so nothing is confounded by encoder differences); the
person subset is split 30/70 into **tune / eval**, CC's α is chosen on **tune** and every system is
reported on the disjoint **eval** split (N=7,531); `PYTHONHASHSEED=0` for bit-exact fusion tie-breaks;
significance by **paired bootstrap** over questions (B=10,000). RRF has no free parameter, so scoring
CC at its best-on-the-test-set α — which an earlier version of this study did — is an oracle that
silently favors CC. (In the event, held-out α picked the *same* value as the oracle in all six cases,
so the inflation here was **+0.0**; the flaw was real but empirically inert. Both are logged.)

### Result 1 — fine-tuned encoder (in-domain dense arm)

| System | top-1 | top-5 | top-20 | top-100 | lexical engine? |
|---|---|---|---|---|---|
| BM25 | 40.7 | 59.5 | 70.8 | 80.0 | yes |
| BM25 + exact-boost | 44.7 | 63.7 | 73.9 | 81.3 | yes |
| **DPR-ft global** — *the engine you start with* | 47.7 | 64.7 | **75.0** | 84.3 | **no** |
| DPR-ft entity-masked, alone | 54.3 | 65.1 | 68.6 | 69.8 | **no** |
| Hybrid RRF (Bruch) | 57.3 | 73.2 | 82.2 | 88.6 | yes |
| Hybrid CC (Bruch, α=0.8 default) | 54.1 | 68.8 | 78.2 | 85.9 | yes |
| Hybrid CC (α=0.5, held-out) | 59.4 | 73.4 | 82.1 | 88.1 | yes |
| Hybrid + exact-boost, RRF | 58.6 | 74.0 | **83.1** | 88.9 | yes |
| Hybrid + exact-boost, CC (α=0.5) | **61.0** | 75.1 | **83.1** | 88.4 | yes |
| **Entity-masked ⊕ global, RRF (ours)** | 59.6 | **75.2** | **83.1** | 88.8 | **no** |
| Entity-masked ⊕ global, CC (α=0.5) | 59.6 | 74.8 | 82.5 | 88.2 | **no** |
| 3-way RRF (BM25 ⊕ global ⊕ masked) | 59.5 | 74.7 | 82.9 | 89.0 | yes |

**With a good dense encoder, the entity mask completely substitutes for a lexical engine.**

| Paired bootstrap, top-20, n=7,531 | Δ | 95% CI | p |
|---|---|---|---|
| entity-masked (ours) **vs dense alone** | **+8.07** | [+7.38, +8.78] | <0.001 |
| entity-masked (ours) vs plain BM25 hybrid | +0.94 | [+0.44, +1.43] | <0.001 |
| **BM25+exact hybrid vs entity-masked (ours)** | **+0.04** | [−0.40, +0.49] | **0.89 — n.s.** |
| 3-way vs entity-masked | −0.21 | [−0.66, +0.24] | 0.38 — n.s. |

The entity mask takes the semantic engine from **75.0 → 83.1 top-20 (+8.1)** and from **47.7 → 59.6
top-1 (+11.9)** — and lands in a **statistical dead heat with the best lexical hybrid** (83.1 vs 83.1,
p=0.89), while never leaving the dense engine. Adding BM25 *on top* of it (the 3-way row) buys nothing.

### Result 2 — zero-shot NQ encoder (weak dense arm)

| System | top-1 | top-5 | top-20 | top-100 | lexical engine? |
|---|---|---|---|---|---|
| BM25 | 40.7 | 59.5 | 70.8 | 80.0 | yes |
| BM25 + exact-boost | 44.7 | 63.7 | 73.9 | 81.3 | yes |
| **DPR-nq global** — *the engine you start with* | 15.0 | 28.0 | **40.8** | 57.8 | **no** |
| DPR-nq entity-masked, alone | 34.7 | 57.2 | 65.7 | 69.1 | **no** |
| Hybrid RRF (Bruch) | 36.7 | 59.5 | 72.4 | 82.3 | yes |
| Hybrid CC (Bruch, α=0.8 default) | 19.3 | 33.5 | **46.7** | 62.8 | yes |
| Hybrid CC (α=0.4, held-out) | 43.3 | 62.4 | 72.8 | 82.4 | yes |
| Hybrid + exact-boost, RRF | 37.7 | 61.9 | 75.0 | 83.6 | yes |
| Hybrid + exact-boost, CC (α=0.4) | 44.2 | 65.1 | 75.5 | 83.2 | yes |
| **Entity-masked ⊕ global, RRF (ours)** | 37.3 | 60.2 | 72.6 | 81.9 | **no** |
| Entity-masked ⊕ global, CC (α=0.1) | 37.3 | 61.8 | 73.3 | 81.4 | **no** |
| **3-way RRF (BM25 ⊕ global ⊕ masked)** | 43.9 | **66.2** | **76.6** | 84.1 | yes |

| Paired bootstrap, top-20, n=7,531 | Δ | 95% CI | p |
|---|---|---|---|
| entity-masked (ours) **vs dense alone** | **+31.79** | [+30.70, +32.88] | <0.001 |
| entity-masked (ours) vs plain BM25 hybrid | +0.17 | [−0.52, +0.88] | 0.67 — n.s. |
| BM25+exact hybrid vs entity-masked (ours) | +2.38 | [+1.75, +3.01] | <0.001 |
| **3-way vs BM25+exact hybrid** | **+1.63** | [+1.23, +2.04] | <0.001 |

**With a weak dense encoder the mask still does most of the work — but a lexical arm now adds real,
orthogonal information.** The mask lifts the engine **40.8 → 72.6 (+31.8)**, which *ties* a plain BM25
hybrid (72.4, p=0.67) — a semantic engine plus a bitmap matching a full lexical⊕semantic hybrid. But
here, unlike the fine-tuned case, adding BM25 on top helps: the **3-way fusion is the best system in
the study (76.6)**, significantly beating every 2-way (+1.63 over the next best).

### The unifying finding

**The entity mask and BM25 are redundant when the dense encoder is good, and complementary when it
isn't.**

- Fine-tuned arm (strong): mask ≡ BM25 hybrid (p=0.89), and 3-way adds nothing (p=0.38).
- Zero-shot arm (weak): mask ≡ plain BM25 hybrid (p=0.67), but 3-way beats both (p<0.001).

The reason is that **BM25 is itself an entity filter** — statistically *and* computationally. It beats
name-filtered dense retrieval outright (70.8 vs 68.6 / 65.7 top-20), and, as the cost model below
shows, its rarest query term usually *is* the entity token. When the dense arm is strong enough to
exploit the entity signal too, the mask and BM25 are two routes to the same information. When the
dense arm is weak, they decorrelate and stack.

### Cost model: what each arm actually touches

Measured on 300 person questions against the real 21M-passage index:

| Per query | Postings / vectors touched | of corpus |
|---|---|---|
| BM25 arm, naive disjunctive (union of query-term postings) | 4,217,681 (median) | 20.1% |
| BM25 arm, block-max WAND floor (rarest term's postings) | 706 (median) | 0.003% |
| **Entity bitmap (ours)** | ~1,000 capped; **118 verified** (mean) | 0.005% |
| Global dense arm (**both** methods pay this**)** | 21,015,324 vectors | 100% |
| Entity-masked dense arm | **118** vectors | 0.0006% |

Two honest readings of this table:

- **We cannot claim a lexical-side speedup.** The rarest term in *"what kind of work does Yehuda
  Amichai do"* **is the entity token**, so a block-max WAND implementation already prunes to roughly
  the same ~10³ documents our bitmap selects. The bitmap largely re-derives what WAND's rarest-term
  pruning does for free. This is the *same* finding as the accuracy result, reached from a different
  direction.
- **The dense arm dominates, and neither method escapes it.** The masked arm is ~178,000× cheaper than
  the global scan (118 vs 21M vectors) — but it's a rounding error beside a global scan we still need,
  since the masked arm *alone* is worse than BM25 (68.6 / 65.7 vs 70.8). The saving is not in FLOPs; it
  is in **not operating a second retrieval system**.

This is a cost *model* from measured postings/vector counts — **not** a latency benchmark. No speedup
is claimed until it is measured end-to-end with proper replication.

### CC vs RRF (what we can say about Bruch's claim)

- **Untuned, RRF wins.** At a fixed α=0.5, CC loses to RRF on both encoders (ft 82.1 vs 82.2; nq 72.0
  vs 72.4). RRF is parameter-free; CC has to be tuned to compete.
- **Tuned, CC wins only when the arms are quality-imbalanced.** On the weak nq encoder, tuned CC beats
  RRF at low k (top-1 **43.3 vs 36.7**) because α can downweight the near-useless dense arm (α*≈0.1–0.4).
  On the balanced ft encoder the two tie (α*=0.5).
- **Bruch's default α=0.8 is catastrophic on a weak dense arm: 46.7 top-20 vs RRF's 72.4.** The single
  most transferable warning in this study is *do not copy a fusion hyperparameter across encoders* —
  α must track the relative quality of the arms.
- The α-curve is flat near its peak (0.4–0.6 within ~1 pt on ft), so tuned CC isn't fragile; it simply
  doesn't earn its parameter unless the arms are lopsided.

### Scope limits (read before quoting any of this)

- **If you already run BM25, the entity mask is redundant** on a strong encoder (+0.9 top-20 over a
  plain hybrid, and 0.0 vs an exact-boosted one). This work is for people who *don't* want a lexical
  engine, or whose dense arm is weak enough that the 3-way fusion pays.
- **The mask is not free: it needs query-side NER** (GLiNER, tens of ms) **and a passage-side
  entity→docid index.** That index is much lighter than a full BM25 index (names only; no term
  frequencies, no norms, no scoring machinery) — but ours was *built from* a Lucene positional index,
  so the standalone version is asserted, not demonstrated.
- **The mask fails silently on 12.5% of questions** (no candidate passage: NER miss, alias, regnal or
  partial name). There it contributes nothing and you fall back to plain dense. BM25 degrades
  gracefully in those cases; the mask does not. This is the method's real weakness and the first thing
  to fix (see [`TODO.md`](TODO.md)).
- Person entities only, one dataset, one corpus. The fine-tuned encoder is trained on *our*
  distant-supervision data, not the paper's exact setup.

> **Comparability note.** §7's tables are the eval split (n=7,531) of one bit-reproducible run per
> encoder; §6/§6b report the full subset (N=10,758) from earlier runs, so cells differ by ≲1 pt. §7 is
> the fair head-to-head (one encoder, all systems, same questions, significance tested) and is the one
> to quote. Per-question first-hit ranks are dumped to `hybrid_{ft,nq}_firsthit.npz`, so any further
> statistic (per-relation CIs, other splits) costs no GPU time.

---

## Limitations & honesty

- **The 300-sample is all P106** (alphabetically first relation), which is favorable for the name
  filter (the subject's own passage reliably contains both the name and the occupation answer).
  §7's held-out numbers are the ones to trust.
- **The entity mask needs query-side NER and a passage-side entity index.** It is not free, and our
  entity index was built *from* a Lucene positional index — so "no lexical engine needed" is
  demonstrated for *scoring*, but the standalone entity-index build is asserted, not demonstrated.
- **The cost model in §7 is a model, not a benchmark.** No latency claim is made; a block-max WAND
  BM25 likely prunes to the same order of magnitude as our bitmap.
- **Throughput figures are observed from the actual runs, not controlled benchmarks** (no warmup /
  replication / variance control). Encoding was partly CPU-tokenization-bound, so the GPU was not
  fully exercised.
- **Distant-supervision training data** is our construction, not the paper's exact Table 2 setup,
  so the fine-tuned numbers reproduce the *finding* (fine-tuning overtakes BM25) but aren't a
  bit-exact replication.
- **Zero-candidate questions remain** (~8–13%): GLiNER false-positive "names", and name forms that
  differ from the corpus beyond diacritics. RRF's global-NN arm mitigates but doesn't eliminate this.

---

## Script index (story order)

| Script | Step | Purpose |
|---|---|---|
| `run_gliner.py`, `run_gliner_large.py` | §1 | person NER over the question set |
| `bm25_retrieve.py` | §2 | BM25 retrieval (current Pyserini API) |
| `dpr_common.py` | §3 | load canonical DPR `.cp` into HF BERT encoders ([CLS] rep) |
| `encode_passages.py`, `encode_questions.py` | §3 | encode corpus & questions (TAG=nq/ft) |
| `dense_search.py`, `dense_eval.py` | §3 | exact GPU streaming search + recall eval |
| `build_train_data.py`, `train_dpr.py` | §4 | distant-supervision data + bi-encoder fine-tune |
| `filtered_dpr.py` | §5 | entity-constrained dense retrieval |
| `fold_util.py`, `fold_shards.py` | §6 | diacritic folding + folded re-index |
| `filtered_rrf.py` | §6 | RRF fusion (name-filter + global NN) with exact tie-break |
| `hybrid_fusion.py` | §7 | all 13 systems: Bruch hybrid, exact-boost, entity-mask, 3-way; RRF vs CC; held-out alpha; per-question first-hit dump |
| `bootstrap_sig.py` | §7 | paired bootstrap over questions (reads the first-hit dump; no GPU) |
| `postings_cost.py` | §7 | cost model: postings/vectors each arm touches |
| `*_driver.sh` | — | orchestration wrappers for the long-running stages |

All numbers in this writeup were measured on the real 21M-passage `psgs_w100` corpus on one RTX 5090.
