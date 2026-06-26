"""Adapted from bm25/bm25_retriever.py for current pyserini (LuceneSearcher API).
Uses Pyserini default BM25 (k1=0.9, b=0.4) — the repo's 'default settings'."""
import os, sys, glob, json, argparse, time
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
REPO = "${REPO}"
sys.path.insert(0, REPO)
from pyserini.search.lucene import LuceneSearcher
import utils.ion as ion
from utils.has_answer_fn import HAS_ANS_FNS

def get_raw(searcher, hit):
    raw = getattr(hit, "raw", None)
    if raw is None:
        d = searcher.doc(hit.docid)
        raw = d.raw() if d is not None else None
    return raw

def search(dataset, n_docs, has_answer_fn, searcher, pid2title):
    results = []
    for entry in dataset:
        question = entry['question']
        answer_lst = entry['answers']
        hits = searcher.search(question, k=n_docs)
        ctxs = []
        for hit in hits:
            title = pid2title[hit.docid]
            contents = json.loads(get_raw(searcher, hit))['contents']
            ctxs.append({'id': hit.docid, 'title': title,
                         'text': contents[len(title):].strip(), 'score': hit.score})
        ctxs = [{**c, 'has_answer': has_answer_fn(c, answer_lst)} for c in ctxs]
        results.append({'question': question, 'answers': answer_lst, 'ctxs': ctxs})
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index_path', required=True)
    p.add_argument('--passage_id_to_title_path', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--glob', action='store_true')
    p.add_argument('--n_docs', type=int, default=100)
    p.add_argument('--answer_type', default='string')
    args = p.parse_args()

    qa_files = sorted(glob.glob(args.input)) if args.glob else [args.input]
    has_answer_fn = HAS_ANS_FNS[args.answer_type]
    pid2title = ion.read_json(args.passage_id_to_title_path, log=True)
    searcher = LuceneSearcher(args.index_path)
    os.makedirs(args.output_dir, exist_ok=True)
    for qa_file in qa_files:
        t0 = time.time()
        dataset = ion.read_json(qa_file)
        results = search(dataset, args.n_docs, has_answer_fn, searcher, pid2title)
        outfile = os.path.join(args.output_dir, os.path.basename(qa_file))
        ion.write_json(outfile, results, pretty=True)
        print(f"  {os.path.basename(qa_file)}: {len(dataset)} qs in {time.time()-t0:.0f}s -> {outfile}")

if __name__ == '__main__':
    main()
