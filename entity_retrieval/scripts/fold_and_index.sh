#!/bin/bash
set -eo pipefail
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
SP=${WORKDIR}
cd $SP
echo "[$(date +%T)] FOLD SHARDS"; python3 fold_shards.py
echo "[$(date +%T)] INDEX FOLDED"
python3 -m pyserini.index.lucene --collection JsonCollection \
  --generator DefaultLuceneDocumentGenerator --threads 8 \
  --input $SP/bm25_shards_folded/ --index $SP/bm25_index_folded/ \
  --storePositions --storeDocvectors --storeRaw 2>&1 | grep -E "documents indexed|Indexing Complete|Total"
echo "[$(date +%T)] FOLDED INDEX DONE"
