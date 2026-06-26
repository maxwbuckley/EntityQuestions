"""Evaluate DPR-NQ dense top-k recall per relation, macro-average, vs paper."""
import os, sys, json, time
import numpy as np
REPO = "${REPO}"
sys.path.insert(0, REPO)
from utils.has_answer_fn import string_match
from utils.tokenizers import SimpleTokenizer

TAG = os.environ.get("TAG", "nq")
SP = "${WORKDIR}"
tok = SimpleTokenizer()

topk_pids = np.load(f"{SP}/dpr_{TAG}_topk_pids.npy", allow_pickle=True)   # (Nq,100) pid strings
meta = json.load(open(f"{SP}/dpr_{TAG}_question_meta.json"))
assert len(meta) == topk_pids.shape[0]

# Gather passage text for needed pids by one scan of the TSV
needed = set(p for row in topk_pids for p in row)
print(f"need text for {len(needed)} unique passages")
pid2text = {}
t0 = time.time()
with open(f"{SP}/psgs_w100.tsv") as f:
    f.readline()
    for line in f:
        i = line.find('\t')
        pid = line[:i]
        if pid in needed:
            parts = line.rstrip('\n').split('\t')
            text = parts[1].strip()
            if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                text = text[1:-1]
            pid2text[pid] = text
            if len(pid2text) == len(needed):
                break
print(f"loaded {len(pid2text)} texts in {time.time()-t0:.0f}s")

K_VALUES = [1, 5, 20, 100]
from collections import defaultdict
rel_total = defaultdict(int)
rel_hit = defaultdict(lambda: defaultdict(int))   # rel -> k -> count
for qi, m in enumerate(meta):
    rel = m["relation"]; answers = m["answers"]
    rel_total[rel] += 1
    found = -1
    for rank, pid in enumerate(topk_pids[qi]):
        ctx = {"text": pid2text.get(pid, "")}
        if string_match(ctx, answers, tok):
            found = rank; break
    for k in K_VALUES:
        if 0 <= found < k:
            rel_hit[rel][k] += 1

paper = json.load(open(f"{SP}/paper_bm25_top20.json"))  # reuse for relation order
tmpl = json.load(open(f"{REPO}/relation_query_templates.json"))
rels = sorted(rel_total)
print(f"\n{'Rel':5} {'Template':38} " + " ".join(f"top{k:>3}" for k in K_VALUES))
print("-"*78)
macro = {k: 0.0 for k in K_VALUES}
for rel in rels:
    accs = {k: 100*rel_hit[rel][k]/rel_total[rel] for k in K_VALUES}
    for k in K_VALUES: macro[k] += accs[k]
    print(f"{rel:5} {tmpl.get(rel,'?')[:36]:38} " + " ".join(f"{accs[k]:5.1f}" for k in K_VALUES))
for k in K_VALUES: macro[k] /= len(rels)
print("-"*78)
print(f"{'MACRO':5} {'':38} " + " ".join(f"{macro[k]:5.1f}" for k in K_VALUES))
ref = "Paper DPR-NQ (zero-shot) macro top-20 = 49.7%" if TAG == "nq" else "Paper: DPR fine-tuned on EntityQuestions improves sharply (Table 2)"
print(f"\n[{TAG}] mine macro top-20 = {macro[20]:.1f}%   |   {ref}")
json.dump({"tag": TAG, "macro": macro, "per_relation": {r: {k: 100*rel_hit[r][k]/rel_total[r] for k in K_VALUES} for r in rels}},
          open(f"{SP}/dpr_{TAG}_results.json", "w"), indent=2)
