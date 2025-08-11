# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List

from summarizer import load_model, summarize

app = FastAPI(title="Bangla News Summarizer", version="1.0")


# ---------- Schemas ----------
class SummarizeReq(BaseModel):
    text: str
    title: Optional[str] = None
    # decoding / control (all optional; summarizer has smart defaults)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_beams: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    length_penalty: Optional[float] = None
    # high-level presets
    mode: Optional[str] = "balanced"   # "compact" | "balanced" | "detailed"
    style: Optional[str] = None        # set "headline" for short titles

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "ঢাকায় আজ বৃষ্টির কারণে সড়কে জলজট সৃষ্টি হয়েছে। প্রশাসন নাগরিকদের সতর্ক থাকতে বলেছে।",
                "num_beams": 4,
                "mode": "compact",
                "style": None
            }
        }
    }


class BatchReq(BaseModel):
    texts: List[str]
    titles: Optional[List[Optional[str]]] = None  # same length as texts or omit
    mode: Optional[str] = "balanced"
    style: Optional[str] = None


# ---------- Lifespan ----------
@app.on_event("startup")
def _startup():
    load_model()  # loads base csebuetnlp/banglat5 (GPU if available)


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
def summarize_api(req: SummarizeReq):
    s = summarize(
        text=req.text,
        title=req.title,
        max_length=req.max_length,
        min_length=req.min_length,
        top_p=req.top_p,
        top_k=req.top_k,
        num_beams=req.num_beams,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        length_penalty=req.length_penalty,
        mode=req.mode,
        style=req.style,
    )
    return {"summary": s}


@app.post("/summarize/batch")
def summarize_batch(req: BatchReq):
    titles = req.titles or [None] * len(req.texts)
    if len(titles) != len(req.texts):
        return {"error": "titles length must match texts length or be omitted"}
    outs = []
    for t, tl in zip(req.texts, titles):
        outs.append(summarize(text=t, title=tl, mode=req.mode, style=req.style))
    return {"summaries": outs}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <meta charset="utf-8">
    <title>Bangla Summarizer</title>
    <body style="font-family:sans-serif;max-width:900px;margin:32px auto;">
      <h1>Bangla News Summarizer</h1>
      <p><em>Paste article text and click Summarize. Use 'headline' style for a short title.</em></p>

      <label>Mode:</label>
      <select id="mode">
        <option value="balanced" selected>balanced</option>
        <option value="compact">compact</option>
        <option value="detailed">detailed</option>
      </select>
      &nbsp;&nbsp;
      <label>Style:</label>
      <select id="style">
        <option value="">(none)</option>
        <option value="headline">headline</option>
      </select>
      &nbsp;&nbsp;
      <label>Num beams (0=sampling)</label>
      <input id="beams" type="number" min="0" max="8" value="4" style="width:60px"/>

      <br><br>
      <textarea id="text" style="width:100%;height:220px;" placeholder="খবরের পুরো টেক্সট পেস্ট করুন..."></textarea><br>
      <button id="go">Summarize</button>
      <pre id="out" style="background:#111;color:#eee;padding:12px;white-space:pre-wrap;border-radius:8px;"></pre>

      <script>
        const go = document.getElementById('go');
        const out = document.getElementById('out');
        go.onclick = async () => {
          out.textContent = 'Summarizing...';
          const text = document.getElementById('text').value;
          const mode = document.getElementById('mode').value || 'balanced';
          const style = document.getElementById('style').value || null;
          const beams = Number(document.getElementById('beams').value || 0);
          const body = { text, mode, style };
          if (beams > 0) body.num_beams = beams;

          const res = await fetch('/summarize', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify(body)
          });
          const data = await res.json();
          out.textContent = data.summary || JSON.stringify(data);
        };
      </script>
    </body>
    """
