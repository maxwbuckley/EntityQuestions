#!/bin/bash
set -eo pipefail
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TF_ENABLE_ONEDNN_OPTS=0
SP=${WORKDIR}
cd $SP
F(){ grep --line-buffered -vE "Loading|warn|UNEXPECTED|^Key|^---|Notes|pooler|absl|oneDNN|port.cc|cpu_feature|To enable"; }
echo "[$(date +%T)] TRAIN DPR";        python3 train_dpr.py 2>&1 | F
echo "[$(date +%T)] RE-ENCODE PASSAGES (ft)"; TAG=ft python3 encode_passages.py 2>&1 | F
echo "[$(date +%T)] ENCODE QUESTIONS (ft)";   TAG=ft python3 encode_questions.py 2>&1 | F
echo "[$(date +%T)] DENSE SEARCH (ft)";       TAG=ft python3 dense_search.py 2>&1 | F
echo "[$(date +%T)] DENSE EVAL (ft)";         TAG=ft python3 dense_eval.py 2>&1 | F
echo "[$(date +%T)] FT PIPELINE DONE"
