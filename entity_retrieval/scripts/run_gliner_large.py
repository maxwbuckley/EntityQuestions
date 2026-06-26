import json, os, glob, time
from gliner import GLiNER

SP = "${WORKDIR}"
TEST_DIR = os.path.join(SP, "dataset", "test")

# Load all test questions, tagged with relation
records = []
for path in sorted(glob.glob(os.path.join(TEST_DIR, "*.test.json"))):
    rel = os.path.basename(path).split(".")[0]
    for item in json.load(open(path)):
        records.append({"relation": rel, "question": item["question"]})
print(f"Loaded {len(records)} questions from {TEST_DIR}")

print("Loading GLiNER model...")
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
model = model.to("cuda")
labels = ["person"]

t0 = time.time()
texts = [r["question"] for r in records]
batch = 64
all_ents = []
for i in range(0, len(texts), batch):
    chunk = texts[i:i+batch]
    preds = model.batch_predict_entities(chunk, labels, threshold=0.5)
    all_ents.extend(preds)
    if (i // batch) % 20 == 0:
        print(f"  {i+len(chunk)}/{len(texts)} ({time.time()-t0:.0f}s)")

for r, ents in zip(records, all_ents):
    persons = [e["text"] for e in ents if e["label"] == "person"]
    r["persons"] = persons
    r["has_person"] = len(persons) > 0

with open(os.path.join(SP, "gliner_results_large.json"), "w") as f:
    json.dump(records, f)

n_total = len(records)
n_person = sum(1 for r in records if r["has_person"])
print(f"\nDONE in {time.time()-t0:.0f}s")
print(f"Questions naming >=1 person: {n_person}/{n_total} ({100*n_person/n_total:.1f}%)")
