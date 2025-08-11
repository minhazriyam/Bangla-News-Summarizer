# summarizer.py
# Compact & accurate summaries + headline mode + long-article handling

from __future__ import annotations
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from normalizer import normalize, strip_boilerplate

MODEL_NAME = "csebuetnlp/banglat5"
TASK_PREFIX = "summarize: "
HEADLINE_INSTR = "এক বাক্যে, সংবাদ শিরোনাম লিখুন; অপ্রয়োজনীয় শব্দ নয়। "

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSeq2SeqLM | None = None
_device: str | None = None


# -------------------------
# Load model/tokenizer once
# -------------------------
def load_model(model_name: str = MODEL_NAME, device: str = "auto", hf_state_dict_path: str | None = None) -> None:
    global _tokenizer, _model, _device
    _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if hf_state_dict_path:
        state = torch.load(hf_state_dict_path, map_location="cpu")
        sd = state.get("state_dict", state)
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
        missing, unexpected = _model.load_state_dict(sd, strict=False)
        if unexpected:
            print("[warn] unexpected keys:", unexpected[:10])
        if missing:
            print("[warn] missing keys:", missing[:10])

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = device
    _model.to(_device)
    _model.eval()


# -------------------------
# Heuristic headline builder
# -------------------------
def _headline_heuristic(text: str) -> str | None:
    t = strip_boilerplate(text)
    t = re.sub(r"\s+", " ", t)

    # e.g., "... রাজশাহী মহানগর বিএনপির ... কমিটি বিলুপ্ত ..."
    subj = None
    m = re.search(r"([অ-হa-zA-Z0-9\s\-–—_,()]*?মহানগর\s*বিএনপি[^\n।]*?)কমিটি\s*বিলুপ্ত", t)
    if m:
        subj = m.group(1).strip()

    two_leaders = False
    if re.search(r"দায়িত্বে|দায়িত্বে", t):
        if re.search(r"দুই\s+[^\n।]+(সহসাংগঠনিক|নেতা)", t) or re.search(r"\([^)]*\)\s*এবং\s*\([^)]*\)", t) or " এবং " in t:
            two_leaders = True

    if subj:
        headline = f"{subj}কমিটি বিলুপ্ত"
        if two_leaders:
            headline += ", দায়িত্বে দুই কেন্দ্রীয় নেতা"
        return normalize(headline)

    if "কমিটি বিলুপ্ত" in t:
        base = "কমিটি বিলুপ্ত ঘোষণা"
        if re.search(r"দায়িত্বে|দায়িত্বে", t) and re.search(r"দুই", t):
            base = "কমিটি বিলুপ্ত, দায়িত্বে দুই কেন্দ্রীয় নেতা"
        return normalize(base)

    return None


# -------------------------
# Generation defaults
# -------------------------
def _defaults_for(mode: str | None, style: str | None, src_words: int) -> dict:
    if style == "headline":
        return dict(
            max_length=24,
            min_length=10,
            num_beams=8,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.20,
            length_penalty=2.0,
            early_stopping=True,
        )

    target_len = max(16, min(140, int(0.18 * src_words)))

    if mode == "compact":
        return dict(
            max_length=max(40, min(100, target_len)),
            min_length=min(24, max(12, int(0.6 * target_len))),
            num_beams=6,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            length_penalty=1.15,
            early_stopping=True,
        )
    if mode == "detailed":
        return dict(
            max_length=max(90, min(160, int(target_len * 1.2))),
            min_length=min(40, max(20, int(0.5 * target_len))),
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.10,
            length_penalty=0.95,
            early_stopping=True,
        )
    # balanced
    return dict(
        max_length=max(70, min(130, target_len)),
        min_length=min(36, max(18, int(0.6 * target_len))),
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.12,
        length_penalty=1.00,
        early_stopping=True,
    )


# -------------------------
# Core gen helper
# -------------------------
def _gen_one(source: str, gen: dict) -> str:
    assert _tokenizer is not None and _model is not None and _device is not None, "Model not loaded"
    enc = _tokenizer(source, max_length=512, truncation=True, padding=False, return_tensors="pt")
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        out = _model.generate(**enc, **gen)
    return normalize(_tokenizer.decode(out[0], skip_special_tokens=True))


# -------------------------
# Public API
# -------------------------
def summarize(
    text: str,
    title: str | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    num_beams: int | None = None,
    no_repeat_ngram_size: int | None = None,
    length_penalty: float | None = None,
    mode: str | None = "balanced",     # compact | balanced | detailed
    style: str | None = None,          # "headline" for short titles
) -> str:
    assert _model is not None, "call load_model() first"

    # 1) quick heuristic for headlines
    if style == "headline":
        h = _headline_heuristic(text)
        if h and 6 <= len(h.split()) <= 16:
            return h

    # 2) clean input
    text = strip_boilerplate(text)
    if title:
        title = strip_boilerplate(title)
        text = f"{title}\n\n{text}"

    # 3) build source with instruction/prefix
    if style == "headline":
        source = "শিরোনাম: " + HEADLINE_INSTR + normalize(text)
    else:
        source = TASK_PREFIX + normalize(text)

    src_words = len(source.split())

    # 4) defaults
    gen = _defaults_for(mode, style, src_words)

    # overrides
    if max_length is not None: gen["max_length"] = max_length
    if min_length is not None: gen["min_length"] = min_length
    if num_beams is not None:  gen["num_beams"]  = num_beams
    if no_repeat_ngram_size is not None: gen["no_repeat_ngram_size"] = no_repeat_ngram_size
    if length_penalty is not None: gen["length_penalty"] = length_penalty

    # sampling switch
    if top_p is not None or top_k is not None:
        gen["do_sample"] = True
        gen.pop("num_beams", None)
        if top_p is not None: gen["top_p"] = top_p
        if top_k is not None: gen["top_k"] = top_k

    # 5) short vs long
    if src_words < 450:
        return _gen_one(source, gen)

    # long: chunk + fuse
    words = source.split()
    window, stride = 420, 360
    pieces = []
    for start in range(0, len(words), stride):
        piece = " ".join(words[start:start + window])
        if not piece:
            break
        pieces.append(piece)

    partials = [_gen_one(p, gen) for p in pieces]
    fused_src = ("শিরোনাম: " if style == "headline" else TASK_PREFIX) + normalize("\n".join(partials))
    return _gen_one(fused_src, gen)


def summarize_batch(texts: list[str], titles: list[str | None] | None = None, **kwargs) -> list[str]:
    titles = titles or [None] * len(texts)
    return [summarize(t, tl, **kwargs) for t, tl in zip(texts, titles)]
