"""DPR-style bi-encoder fine-tuning on EntityQuestions train data.
Init from the DPR-NQ .cp encoders; in-batch + hard-negative NLL loss; [CLS] reps.
Saves fine-tuned question/ctx encoder state dicts."""
import os, sys, json, time, math, random
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "${WORKDIR}")
from dpr_common import load_dpr_encoders

SP = "${WORKDIR}"
BATCH = 32
H = 1            # hard negatives per example used in loss
EPOCHS = 10
LR = 2e-5
MAXLEN_Q, MAXLEN_C = 64, 256
dev = "cuda"

data = json.load(open(f"{SP}/dpr_train_data.json"))
print(f"train examples: {len(data)}")

q_enc, c_enc, tok = load_dpr_encoders(device=dev, half=False)  # trainable fp32
q_enc.train(); c_enc.train()

class DS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        e = self.d[i]
        pos = e["positive_ctxs"][0]
        hns = e["hard_negative_ctxs"][:H]
        while len(hns) < H: hns.append(hns[-1])
        return e["question"], pos, hns

def collate(batch):
    qs = [b[0] for b in batch]
    ctx_titles, ctx_texts = [], []
    for _, pos, hns in batch:
        ctx_titles.append(pos["title"]); ctx_texts.append(pos["text"])
        for hn in hns:
            ctx_titles.append(hn["title"]); ctx_texts.append(hn["text"])
    qf = tok(qs, padding=True, truncation=True, max_length=MAXLEN_Q, return_tensors="pt")
    cf = tok(ctx_titles, ctx_texts, padding=True, truncation=True, max_length=MAXLEN_C, return_tensors="pt")
    return qf, cf

dl = DataLoader(DS(data), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)
opt = torch.optim.AdamW(list(q_enc.parameters()) + list(c_enc.parameters()), lr=LR, weight_decay=0.0)
total_steps = len(dl) * EPOCHS
warmup = int(0.06 * total_steps)
def lr_lambda(s): return min(1.0, s/max(1,warmup)) * max(0.0, (total_steps-s)/max(1,total_steps-warmup))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
lossf = torch.nn.CrossEntropyLoss()

t0 = time.time(); step = 0
for ep in range(EPOCHS):
    run = 0.0; correct = 0; seen = 0
    for qf, cf in dl:
        qf = {k: v.to(dev) for k, v in qf.items()}; cf = {k: v.to(dev) for k, v in cf.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qemb = q_enc(**qf).last_hidden_state[:, 0]          # (B,d)
            cemb = c_enc(**cf).last_hidden_state[:, 0]          # (B*(1+H),d)
            scores = qemb @ cemb.t()                            # (B, B*(1+H))
            target = torch.arange(qemb.size(0), device=dev) * (1 + H)
            loss = lossf(scores.float(), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q_enc.parameters())+list(c_enc.parameters()), 2.0)
        opt.step(); sched.step(); step += 1
        run += loss.item(); seen += qemb.size(0)
        correct += (scores.argmax(1) == target).sum().item()
        if step % 100 == 0:
            print(f"  ep{ep} step{step}/{total_steps} loss{run/100:.3f} acc{correct/seen:.3f} "
                  f"lr{sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
            run = 0.0; correct = 0; seen = 0
    print(f"[epoch {ep} done] {time.time()-t0:.0f}s", flush=True)

os.makedirs(f"{SP}/dpr_ft", exist_ok=True)
torch.save(q_enc.state_dict(), f"{SP}/dpr_ft/q_encoder.pt")
torch.save(c_enc.state_dict(), f"{SP}/dpr_ft/c_encoder.pt")
print(f"DONE training in {(time.time()-t0)/60:.1f} min; saved to {SP}/dpr_ft/")
