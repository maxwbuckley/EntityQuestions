"""Fold diacritics in the BM25 passage contents and write new shards for indexing."""
import os, sys, json, glob, time
sys.path.insert(0, "${WORKDIR}")
from fold_util import fold

SP = "${WORKDIR}"
SRC = f"{SP}/bm25_shards"
DST = f"{SP}/bm25_shards_folded"; os.makedirs(DST, exist_ok=True)

t0 = time.time()
for path in sorted(glob.glob(f"{SRC}/shard*.json")):
    name = os.path.basename(path)
    recs = json.load(open(path))
    for r in recs:
        r["contents"] = fold(r["contents"])
    json.dump(recs, open(f"{DST}/{name}", "w"))
    print(f"  folded {name}: {len(recs)} recs ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE folding in {(time.time()-t0)/60:.1f} min")
