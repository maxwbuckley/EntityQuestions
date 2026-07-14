# Entity-Centric Retrieval: Reproducing EntityQuestions and Beating BM25 with Entity-Constrained Deep Hybrid Retrieval

This directory documents an end-to-end study built on top of the
[EntityQuestions](https://arxiv.org/pdf/2109.08535.pdf) dataset (Sciavolino et al., EMNLP 2021).
It starts by **reproducing the paper's central result** — that BM25 beats dense retrieval (DPR)
on simple entity-centric questions — and then asks a follow-up question:

> DPR is bad at entity questions. Is that because its vectors *rank* badly, or because it
> *retrieves the wrong entity* out of 21M passages? And if it's the latter, can we fix it by
> hard-constraining retrieval to passages that actually mention the entity?

The answer turns out to be a clean, useful finding: **DPR's weakness on entity questions is
mostly a retrieval (recall) problem, not a ranking problem.** If you hard-filter the corpus to
passages that mention the question's named entity (using the inverted index), then rank only
those with DPR vectors, and fuse that with global dense nearest-neighbors via Reciprocal Rank
Fusion (RRF), you get a retriever that **decisively beats BM25** on person questions — especially
when the dense encoder is fine-tuned in-domain.

> **Read §7 before quoting the headline.** A critical baseline (§7) shows that a *plain* BM25⊕DPR hybrid
> with **no entity filter** already matches the entity-constrained method — on **both** encoders (82.0
> vs 82.8 top-20 fine-tuned; 72.7 vs 72.8 zero-shot). The reason: **BM25 is already the entity-precision
> signal**, so an entity-constrained *dense* retriever is largely redundant with standard hybrid
> retrieval. The entity filter's only measurable edge is +1.9 top-1 on the fine-tuned encoder, and it
> vanishes on the zero-shot one. The honest contribution is the **diagnosis** (§5: dense-alone fails to
> *retrieve* the entity), not "entity filtering beats BM25." See §7/§7b.

Everything here was run against the real DPR Wikipedia corpus (`psgs_w100`, 21,015,324 passages)
on a single RTX 5090.

---

## Headline results

**Full test set (24 relations, 22,075 questions), macro-averaged top-20 retrieval accuracy:**

| Retriever | top-20 | notes |
|---|---|---|
| BM25 | **72.0%** | exact reproduction of the paper (Table 1 / Table 5) |
| DPR-NQ (zero-shot) | **50.9%** | reproduces the paper's ~49.7% (DPR ≪ BM25, the paper's thesis) |
| DPR fine-tuned on EntityQuestions train | **76.2%** | in-domain fine-tuning overtakes BM25 (paper Table 2) |

**Person-question subset — the full, honest panel (all 10,758 person questions, fine-tuned encoder,
one shared encoder across every row so the comparison is fair). Details in §7:**

| System | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|
| BM25 | 41.2 | 59.8 | 71.2 | 80.2 |
| **BM25 + exact-name boost** (lexical only, **no dense model**) | **45.1** | **64.0** | **74.1** | **81.5** |
| DPR-ft (global dense) | 47.9 | 64.2 | 74.7 | 84.2 |
| DPR-ft name-filtered (entity-constrained dense) | 54.6 | 65.5 | 68.9 | 70.2 |
| Generic hybrid BM25 ⊕ DPR-ft, **RRF** (Bruch-style, no entity filter) | 57.7 | 73.3 | 82.0 | 88.4 |
| Entity-constrained hybrid, RRF (**ours**) | **59.6** | **75.1** | **82.8** | **88.6** |

Two things to read off this table before anything else:

- **BM25 + exact-name boost is the cheapest win in the study: +3.9 top-1 / +2.9 top-20 over plain
  BM25, for zero dense compute.** All it does is float passages containing the exact detected person
  name above the rest of the BM25 list. It is *encoder-independent* — the same +2.9 shows up on the
  zero-shot panel (§7b) — and it beats *entity-constrained dense retrieval* at top-20 (74.1 vs 68.9).
  If you take one thing from this repo into a production lexical stack, take this.
- **Hybridization, not the entity filter, is what beats BM25.** The plain BM25⊕DPR hybrid gets 82.0;
  adding our entity constraint moves it to 82.8 (+0.8). See the caveat above and §7/§7b.

The **300-question P106 sample** used during development appears in §5–§6; it is one easy relation
and overstates the name-filter, so always quote the full subset above. The **zero-shot NQ encoder**
panel of the same table is in §7b.

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
  _(But see §7: a plain BM25⊕DPR hybrid with no entity filter also reaches ~82 top-20, so most of that
  +11.2 is hybridization, not the entity constraint. The entity filter's real contribution is a low-k
  precision edge.)_
- **The name filter contributes precision the dense model lacks:** at top-1 it beats full dense
  retrieval for both encoders (NQ 15.1→35.3; ft 47.9→55.0), and the fusion improves on both
  (ft top-1: full 47.9 → filtered 55.0 → fused 60.0).
- Numbers are lower than the P106-only 300-sample (e.g. ft top-20 92.3 → 82.4), as expected — but
  the ordering (RRF > full > name-filter, RRF > BM25) is unchanged.
- Diacritic folding moved the zero-candidate rate only 12.9%→12.7% on the full set: the residual is
  dominated by GLiNER false-positive "names" and non-diacritic name-form mismatches, not accents.
  See `HANDOFF.md` for why this matters and what to do about it.

---

## 7. The critical baseline: is it the *entity constraint*, or just hybridization? (Bruch et al.)

**Why this section exists.** Sections 5–6 fuse an entity-filtered dense list with global dense
nearest-neighbors and beat BM25. But there is an obvious confound: **maybe any lexical⊕dense hybrid
does that**, and the entity filter adds nothing beyond ordinary hybridization. The only way to know is
to build the standard hybrid — BM25 ⊕ dense — and compare, *on the same encoder*. This is the
experiment [`HANDOFF.md`](HANDOFF.md) flagged as **CRITICAL**: "until this is run the headline claim is
not defensible." Here it is.

The reference for hybrid fusion is **Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid
Retrieval"** (arXiv [2210.11934](https://arxiv.org/abs/2210.11934), ACM TOIS 2023). They fuse a lexical
ranking (BM25) and a semantic ranking (dense) two ways:

- **Reciprocal Rank Fusion (RRF):** `f = 1/(η + π_Lex) + 1/(η + π_Sem)` (rank-based; η=60).
- **Convex Combination (CC):** `f = α·φ_Sem(s_Sem) + (1−α)·φ_Lex(s_Lex)`, where φ is per-query
  **min-max normalization** `φ(s)=(s−m_q)/(M_q−m_q)` (their TM2C2 uses theoretical bounds: BM25 inf=0,
  cosine inf=−1). Their findings: **CC > RRF in- and out-of-domain, RRF is η-sensitive, and CC is
  normalization-agnostic and sample-efficient**, with α≈0.8 a good default.

We add all of these, **computed on the single fine-tuned DPR encoder** so nothing is confounded by
encoder differences (`scripts/hybrid_fusion.py`). For the two CC systems we sweep α∈{0.1,…,0.9} and
report the best top-20 operating point (the full α-curve is logged).

**Result — full person subset (N=10,758), fine-tuned DPR, one shared encoder:**

| System | Fusion | Entity filter | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|---|---|
| BM25 | — | no | 41.2 | 59.8 | 71.2 | 80.2 |
| BM25 + exact-boost | exact-match boost | lexical | 45.1 | 64.0 | 74.1 | 81.5 |
| DPR-ft global | — | no | 47.9 | 64.2 | 74.7 | 84.2 |
| DPR-ft name-filtered | — | yes | 54.6 | 65.5 | 68.9 | 70.2 |
| **Hybrid RRF** (Bruch) | RRF | **no** | 57.7 | 73.3 | **82.0** | 88.4 |
| **Hybrid CC** (Bruch, α\*=0.5) | convex comb. | **no** | 59.4 | 73.6 | 81.9 | 87.9 |
| **Entity RRF** (ours, §6) | RRF | **yes** | 59.6 | 75.1 | **82.8** | 88.6 |
| **Entity CC** (ours, α\*=0.5) | convex comb. | **yes** | 59.8 | 74.7 | 82.2 | 88.0 |

**The honest headline: on the fine-tuned encoder, hybridization explains almost all of the gain over
BM25, and the entity constraint adds only a small low-k edge.**

- **Generic hybrid already gets 82.0 top-20** (BM25 71.2 → 82.0, **+10.8**), with *no entity filter at
  all*. Our entity-constrained fusion gets 82.8 — a further **+0.8**. So the entity constraint is **not**
  the main driver here; ordinary BM25⊕DPR hybridization is.
- **Where the entity filter helps is precision (low k):** Entity-RRF − Hybrid-RRF is **+1.9 top-1,
  +1.8 top-5, +0.8 top-20, +0.2 top-100.** The exact-entity constraint removes near-miss distractors at
  the very top of the ranking; by top-100 the global dense arm already has recall covered, so the gap
  closes. A real but modest *ranking-precision* effect, not a recall one.
- **CC vs RRF (contra Bruch):** here **RRF ≥ CC** for both the generic and entity hybrids (82.0 vs 81.9;
  82.8 vs 82.2). CC only matches RRF after tuning α down to ≈0.5 — Bruch's α=0.8 default is worse on this
  data (top-20 ≈ 78; see the logged α-curve). Bruch established CC>RRF on web/BEIR retrieval; on this
  narrow entity-QA distribution with a strong fine-tuned dense arm, rank-based RRF is at least as good
  and CC's optimal mixing weight shifts toward the lexical/precision arm. The α-curve is flat near its
  peak (0.4–0.6 within ~1 pt), so CC isn't fragile — it just doesn't win here.
- **exact-match-boosted BM25** (a lexical-only entity signal) lifts BM25 71.2 → 74.1 top-20 and 41.2 →
  45.1 top-1 — cheap and real, but far below any hybrid.

**What this means (reframing).** The §6 claim "entity-constrained RRF beats BM25 (≈82 vs 71)" is *true*,
but this baseline shows a plain hybrid gets there too — so the earlier framing credited the entity filter
for gains that mostly belong to hybridization. The defensible contributions are now: **(1)** the
diagnosis of §5 (DPR's entity weakness is retrieval-failure, not ranking-failure); **(2)** a plain
BM25⊕DPR hybrid recovers most of it once the encoder is fine-tuned; **(3)** the entity constraint adds a
small, consistent *low-k precision* gain on top.

**Does the entity filter earn its keep on a *weaker* encoder?** All of the above is on the fine-tuned
encoder, whose dense arm is already strong on entities (74.7 top-20). The natural hypothesis: on the
**zero-shot NQ** encoder, where global DPR is far weaker (41.1 top-20), the entity constraint should
finally matter — its marginal value inversely related to the encoder's entity competence. §7b tests
exactly that, on the same shared-encoder discipline. **It refutes the hypothesis.**

### 7b. The crux, tested on the zero-shot encoder — and refuted

We ran the identical comparison on the **zero-shot DPR-NQ** encoder (`hybrid_fusion.py 0 nq`), same
shared-encoder discipline, same full person subset.

**Result — full person subset (N=10,758), zero-shot DPR-NQ, one shared encoder:**

| System | Fusion | Entity filter | top-1 | top-5 | top-20 | top-100 |
|---|---|---|---|---|---|---|
| BM25 | — | no | 41.2 | 59.8 | 71.2 | 80.2 |
| BM25 + exact-boost | exact-match boost | lexical | 45.1 | 64.0 | 74.1 | 81.5 |
| DPR-nq global | — | no | 15.1 | 28.2 | 41.1 | 58.0 |
| DPR-nq name-filtered | — | yes | 35.1 | 57.4 | 66.2 | 69.5 |
| Hybrid RRF (Bruch) | RRF | no | 37.3 | 60.1 | 72.7 | 82.4 |
| **Hybrid CC** (Bruch, α\*=0.4) | convex comb. | no | 43.9 | 62.7 | 73.1 | 82.6 |
| Entity RRF (ours) | RRF | yes | 37.4 | 60.5 | 72.8 | 81.9 |
| **Entity CC** (ours, α\*=0.1) | convex comb. | yes | 37.4 | 62.0 | **73.5** | 81.4 |

**The hypothesis is refuted: the entity filter still adds essentially nothing over a plain hybrid.**
Entity-RRF − Hybrid-RRF = **+0.1 top-20, +0.1 top-1** (vs +0.8 / +1.9 on the ft encoder). Even when the
dense arm is nearly useless (41.1), constraining it by entity name does not beat the generic BM25⊕DPR
hybrid. The marginal value of the entity filter did **not** rise on the weak encoder.

**Why — the real lesson: BM25 *is* the entity-precision signal.** A generic hybrid already contains a
lexical arm, and BM25 is a *better* exact-entity matcher than filtering dense vectors by name, on **both**
encoders (name-filtered dense 66.2 / 68.9 top-20 vs BM25's 71.2). So "entity-constrained dense retrieval"
is largely **redundant with standard hybrid retrieval**: the precision it adds is precision that
BM25-in-a-hybrid already supplies for free. The §5 diagnosis (dense-alone fails to *retrieve* the entity)
stands; the §6 *method* is mostly subsumed by hybridization. Its only measurable residual edge is +1.9
top-1 on the strong ft encoder — which vanishes here.

**CC vs RRF flips and now favors CC.** On the weak NQ encoder **CC beats RRF** (Hybrid CC 73.1 vs 72.7;
and 43.9 vs 37.3 at top-1) because CC's tunable α can *downweight the near-useless dense arm* (best
α≈0.1–0.4, i.e. lean lexical), whereas rank-based RRF gives the bad arm a fixed contribution. This is
Bruch's flexibility argument, and it bites hardest when the two arms are quality-imbalanced. The catch
that cuts the other way: Bruch's **default α=0.8 is catastrophic here (46.8 top-20)** — it overweights the
useless dense arm — so CC's advantage is entirely contingent on tuning α to arm quality. Net across both
encoders: **with α tuned, CC ≥ RRF; the gap is ~0 when arms are balanced (ft) and clearly positive when
they're imbalanced (nq).**

**Bottom line of §7.** The strongest honest claims from this study: **(1)** [§5] dense retrieval's entity
weakness is a *retrieval* failure, not a ranking one; **(2)** standard BM25⊕DPR hybrid retrieval already
fixes most of it, on **both** a zero-shot and a fine-tuned encoder; **(3)** a bespoke entity-constrained
dense retriever adds little beyond the hybrid, because BM25 already provides the entity precision — its
only edge is +1.9 top-1 on a strong encoder; **(4)** convex-combination fusion earns its extra parameter
mainly when the lexical and dense arms are quality-imbalanced, where it downweights the weak arm and beats
RRF. The obvious untested variant that *could* still rescue the method: a **three-way** fusion
(BM25 ⊕ global-dense ⊕ entity-filtered-dense), to check whether the name filter adds anything *orthogonal*
to BM25 — see [`TODO.md`](TODO.md).

> **Comparability note.** Each table is one self-consistent shared-encoder run (§7 fine-tuned, §7b
> zero-shot NQ), so its
> BM25 / DPR-full / name-filter / Entity-RRF cells differ by ≲1 pt from §6b (which used the original
> checkpoint). §6b is kept as the previously-reported reference; this section is the fair head-to-head
> (one encoder, all systems). CC's α is tuned on this same subset (an oracle upper bound; Bruch show one
> held-out α transfers, so a clean split would cost little — flagged in `TODO.md`). Fusion tie-breaks
> are Python hash-order dependent, so individual cells move ≤0.1 pt run-to-run (set `PYTHONHASHSEED`
> for bit-exact repro); this doesn't affect any comparison here.

---

## Limitations & honesty

- **The 300-sample is all P106** (alphabetically first relation), which is favorable for the name
  filter (the subject's own passage reliably contains both the name and the occupation answer).
  Full-subset numbers (§6b) are the ones to trust.
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
| `hybrid_fusion.py` | §7 | generic hybrid (Bruch) + exact-boost + convex-combination vs RRF |
| `*_driver.sh` | — | orchestration wrappers for the long-running stages |

All numbers in this writeup were measured on the real 21M-passage `psgs_w100` corpus on one RTX 5090.
