#!/bin/bash
set -eo pipefail
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TF_ENABLE_ONEDNN_OPTS=0
SP=${WORKDIR}
cd $SP
echo "[$(date +%T)] encode questions"; python3 encode_questions.py 2>&1 | grep -vE "Loading|warn|UNEXPECTED|^Key|^---|Notes|pooler|absl|oneDNN|port.cc|cpu_feature|To enable"
echo "[$(date +%T)] dense search"; python3 dense_search.py 2>&1 | grep -vE "Loading|warn"
echo "[$(date +%T)] dense eval"; python3 dense_eval.py 2>&1 | grep -vE "Loading|warn"
echo "[$(date +%T)] DENSE PIPELINE DONE"
