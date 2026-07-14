"""Cost model: how many postings does each retrieval arm actually touch?

Bruch-style hybrid: BM25 must score the UNION of the query terms' postings lists.
Ours:               a single bigram/phrase bitmap for the entity name.
Both:               pay for the global dense arm.
"""
import os, sys, json
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
SP="/home/maxwb/entityq_work"; REPO="/mnt/c/Users/maxwb/Development/EntityQuestions"
sys.path.insert(0, REPO); sys.path.insert(0, SP)
import numpy as np
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from pyserini.search.lucene import LuceneSearcher
from fold_util import fold

N_Q  = int(sys.argv[1]) if len(sys.argv)>1 else 300
NDOC = 21015324
ir   = IndexReader(f"{SP}/bm25_index")
sf   = LuceneSearcher(f"{SP}/bm25_index_folded")
meta = json.load(open(f"{SP}/dpr_ft_question_meta.json"))
gl   = json.load(open(f"{SP}/gliner_results_large.json"))
pidx = [i for i,g in enumerate(gl) if g["has_person"]][:N_Q]

union, ent_bitmap, rarest, nterms = [], [], [], []
for qi in pidx:
    q = meta[qi]["question"]
    terms = ir.analyze(q)                      # stemmed, stopworded -- what Lucene actually scores
    dfs = []
    for t in terms:
        try: dfs.append(ir.get_term_counts(t, analyzer=None)[0])   # df
        except Exception: dfs.append(0)
    dfs = [d for d in dfs if d > 0]
    if not dfs: continue
    union.append(sum(dfs))                     # postings a disjunctive scorer must traverse
    rarest.append(min(dfs))                    # what WAND/block-max can prune down toward
    nterms.append(len(dfs))
    # our arm: the entity phrase bitmap
    n_ent = 0
    for nm in gl[qi]["persons"]:
        try: n_ent += len(sf.search(f'"{fold(nm)}"', k=1000))
        except Exception: pass
    ent_bitmap.append(n_ent)

u  = np.array(union); e = np.array(ent_bitmap, dtype=float); r = np.array(rarest)
def s(name, a, unit=""):
    print(f"{name:44} median {np.median(a):12,.0f}   mean {a.mean():12,.0f}   p90 {np.percentile(a,90):12,.0f}{unit}")
print(f"\n=== Postings touched per query (N={len(u)} person questions, corpus {NDOC:,} passages) ===")
s("BM25 arm: UNION of query-term postings",  u)
s("BM25 arm: rarest term's postings (WAND floor)", r)
s("Ours: entity phrase/bigram bitmap (docs)", e)
print(f"\nquery terms after analysis: median {np.median(nterms):.0f}")
print(f"\nratio  union / entity-bitmap        : {np.median(u)/max(np.median(e),1):,.0f}x")
print(f"ratio  corpus / entity-bitmap       : {NDOC/max(np.median(e),1):,.0f}x")
print(f"entity bitmap as % of corpus        : {100*np.median(e)/NDOC:.6f}%")
print(f"union as % of corpus                : {100*np.median(u)/NDOC:.2f}%")
