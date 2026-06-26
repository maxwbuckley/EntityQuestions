"""Build DPR-format fine-tuning data from the EntityQuestions train split via
BM25 distant supervision: positive = top answer-bearing passage, hard negatives =
top non-answer passages. Mirrors how DPR sources positives/hard-negatives."""
import os, sys, glob, json, random, time
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
REPO = "${REPO}"
SP = "${WORKDIR}"
sys.path.insert(0, REPO)
from pyserini.search.lucene import LuceneSearcher
import utils.ion as ion
from utils.has_answer_fn import string_match
from utils.tokenizers import SimpleTokenizer

PER_REL = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
N_HARD = 3
tok = SimpleTokenizer()
pid2title = ion.read_json(f"{SP}/pid2title.json")
searcher = LuceneSearcher(f"{SP}/bm25_index")
searcher.set_bm25(0.9, 0.4)

def raw_text(searcher, docid, title):
    d = searcher.doc(docid)
    contents = json.loads(d.raw())["contents"]
    return contents[len(title):].strip()

random.seed(0)
out = []
t0 = time.time()
for path in sorted(glob.glob(f"{SP}/dataset/train/*.train.json")):
    rel = os.path.basename(path).split(".")[0]
    items = json.load(open(path))
    random.shuffle(items)
    kept = 0
    for it in items:
        if kept >= PER_REL:
            break
        q, answers = it["question"], it["answers"]
        hits = searcher.search(q, k=30)
        pos, hard = None, []
        for h in hits:
            title = pid2title[h.docid]
            text = raw_text(searcher, h.docid, title)
            ctx = {"title": title, "text": text, "passage_id": h.docid}
            if string_match({"text": text}, answers, tok):
                if pos is None:
                    pos = ctx
            else:
                if len(hard) < N_HARD:
                    hard.append(ctx)
        if pos is not None and hard:
            out.append({"question": q, "answers": answers,
                        "positive_ctxs": [pos], "hard_negative_ctxs": hard, "relation": rel})
            kept += 1
    print(f"  {rel}: kept {kept} ({time.time()-t0:.0f}s, total {len(out)})", flush=True)

json.dump(out, open(f"{SP}/dpr_train_data.json", "w"))
print(f"DONE: {len(out)} training examples with positives+hardnegs in {(time.time()-t0)/60:.1f} min")
