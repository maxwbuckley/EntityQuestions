#!/bin/bash
set -eo pipefail
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 TF_ENABLE_ONEDNN_OPTS=0
SP=${WORKDIR}
cd ${REPO}
echo "===== FULL SUBSET: DPR-NQ + RRF ====="
PYTHONPATH=.:$SP python3 $SP/filtered_rrf.py 0 nq 2>&1 | grep -vE "WARNING|INFO|absl|oneDNN|port.cc|warn"
echo "===== FULL SUBSET: DPR-FT + RRF ====="
PYTHONPATH=.:$SP python3 $SP/filtered_rrf.py 0 ft 2>&1 | grep -vE "WARNING|INFO|absl|oneDNN|port.cc|warn"
echo "===== RRF FULL DONE ====="
