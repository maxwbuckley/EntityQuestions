"""Load the canonical DPR-NQ .cp checkpoint (fbaipublicfiles) into HF BertModel
encoders, fully offline. DPR's encoder representation is the [CLS] last_hidden_state
(NOT BERT's tanh pooler), so we encode with BertModel and take last_hidden_state[:, 0]."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import collections, sys, types
import torch
from transformers import BertModel, BertConfig, BertTokenizerFast

CP_PATH = "${WORKDIR}/dpr_nq.cp"

def _register_checkpoint_namedtuple():
    CheckpointState = collections.namedtuple(
        "CheckpointState",
        ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"],
    )
    for path in ["dpr", "dpr.utils", "dpr.utils.model_utils"]:
        sys.modules.setdefault(path, types.ModuleType(path))
    sys.modules["dpr.utils.model_utils"].CheckpointState = CheckpointState

def _load_model_dict(cp_path=CP_PATH):
    _register_checkpoint_namedtuple()
    state = torch.load(cp_path, map_location="cpu", weights_only=False)
    return state.model_dict if hasattr(state, "model_dict") else state["model_dict"]

def _submodel(md, prefix, device, half):
    sd = {}
    for k, v in md.items():
        if k.startswith(prefix):
            nk = k[len(prefix):]
            if nk.startswith("encode_proj"):   # projection head; unused for base model
                continue
            sd[nk] = v
    config = BertConfig.from_pretrained("bert-base-uncased")
    m = BertModel(config, add_pooling_layer=True)
    res = m.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not k.startswith("pooler")]
    unexp = [k for k in res.unexpected_keys if not k.startswith("pooler")]
    assert not miss and not unexp, f"weight mismatch for {prefix}: missing={miss[:5]} unexpected={unexp[:5]}"
    m = m.to(device).eval()
    return m.half() if half else m

def load_dpr_encoders(cp_path=CP_PATH, device="cuda", half=True):
    md = _load_model_dict(cp_path)
    q = _submodel(md, "question_model.", device, half)
    c = _submodel(md, "ctx_model.", device, half)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return q, c, tok

def build_bert(state_dict_path, device, half):
    """Build a BertModel encoder from a saved fine-tuned state dict (.pt)."""
    config = BertConfig.from_pretrained("bert-base-uncased")
    m = BertModel(config, add_pooling_layer=True)
    sd = torch.load(state_dict_path, map_location="cpu")
    res = m.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not k.startswith("pooler")]
    unexp = [k for k in res.unexpected_keys if not k.startswith("pooler")]
    assert not miss and not unexp, f"FT weight mismatch: missing={miss[:5]} unexpected={unexp[:5]}"
    m = m.to(device).eval()
    return m.half() if half else m

def load_tagged_encoders(tag, device="cuda", half=True):
    """tag='nq' -> canonical DPR-NQ from .cp; tag='ft' -> fine-tuned .pt encoders."""
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if tag == "nq":
        q, c, _ = load_dpr_encoders(device=device, half=half)
        return q, c, tok
    base = os.path.dirname(CP_PATH) + "/dpr_ft"
    q = build_bert(f"{base}/q_encoder.pt", device, half)
    c = build_bert(f"{base}/c_encoder.pt", device, half)
    return q, c, tok

@torch.no_grad()
def encode(model, tok, texts, text_pairs=None, device="cuda", max_length=256):
    feats = tok(texts, text_pairs, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt").to(device)
    return model(**feats).last_hidden_state[:, 0].float().cpu()  # DPR uses [CLS]
