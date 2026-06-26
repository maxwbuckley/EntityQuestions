#!/bin/bash
set -e
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
SP=${WORKDIR}
REPO=${REPO}
cd $REPO

echo "[$(date +%T)] STEP 0: verify gzip integrity"
gunzip -t $SP/psgs_w100.tsv.gz && echo "gzip OK"

echo "[$(date +%T)] STEP 1: decompress"
if [ ! -f $SP/psgs_w100.tsv ]; then
  gunzip -k $SP/psgs_w100.tsv.gz
fi
wc -l $SP/psgs_w100.tsv

echo "[$(date +%T)] STEP 2: preprocess into shards (repo script)"
mkdir -p $SP/bm25_shards
PYTHONPATH=$REPO python3 bm25/build_bm25_ctx_passages.py \
  --wiki_passages_file $SP/psgs_w100.tsv \
  --outdir $SP/bm25_shards/ \
  --title_index_path $SP/pid2title.json \
  --n_shards 20

echo "[$(date +%T)] STEP 3: build Lucene BM25 index"
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --input $SP/bm25_shards/ \
  --index $SP/bm25_index/ \
  --storePositions --storeDocvectors --storeRaw

echo "[$(date +%T)] BUILD PIPELINE DONE"
