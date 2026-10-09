"""Puente API and reviewer page.

POST /intake {"text": "...", "pipeline": "direct", "guard": true}  -> English intake + evidence checks + word alerts
POST /transcribe (audio file)                                     -> Spanish text (faster-whisper, if installed)
GET  /                                                             -> reviewer page
Backend: PUENTE_BACKEND=gemini (needs GEMINI_API_KEY), hf (local GPU model in PUENTE_MODEL), or rules
(keyword baseline, no model; used by CI).
Demo only: synthetic data, not for clinical use, no real patient information.
"""
import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from collections import defaultdict, deque

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from puente import pipelines
from puente.lexicon import LEXICON, SOURCES

app = FastAPI(title="Puente", version="1.0",
              description="Spanish patient message -> structured English intake for clinician review. Demo with "
                          "synthetic data only; not for clinical use.")
PUBLIC = Path(__file__).resolve().parent.parent / "public"
api = APIRouter()
RATE = int(os.environ.get("PUENTE_RATE_PER_MIN", "8"))       # per visitor, best effort (per server instance)
_hits = defaultdict(deque)


def rate_limit(request: Request):
    who = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
    now, q = time.time(), _hits[who]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "Too many requests from this address; wait a minute.")
    q.append(now)


class IntakeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000, examples=["Comí mariscos y desde anoche estoy intoxicado."])
    pipeline: Literal["direct", "translate"] = "direct"
    guard: bool = True


@lru_cache(maxsize=1)
def get_llm():
    backend = os.environ.get("PUENTE_BACKEND", "gemini")
    if backend == "gemini":
        from puente.llm import Gemini
        return Gemini(os.environ.get("PUENTE_MODEL", "gemini-3.5-flash-lite"), min_interval=0)
    if backend == "rules":
        from puente.rules import Rules
        return Rules()
    if backend == "hf":
        from puente.llm import LocalHF
        return LocalHF(os.environ.get("PUENTE_MODEL", "Qwen/Qwen2.5-7B-Instruct"), batch=1)
    raise RuntimeError(f"unknown PUENTE_BACKEND {backend}")


def word_alerts(text):
    """Every lexicon pitfall in the message (including the ones the guard does not use), for the reviewer."""
    t = text.lower()
    return [{"family": e["family"], "note": e["note"], "sources": [SOURCES[s] for s in e["sources"]]}
            for e in LEXICON if re.search(e["pattern"], t)]


def evidence_checks(intake, source):
    src = re.sub(r"\s+", " ", source.lower())
    out = []
    for key in ("symptoms", "conditions", "medications", "allergies"):
        for item in intake.get(key, []):
            q = item.get("evidence", "")
            if q:
                out.append({"field": key, "item": item.get("name") or item.get("substance"), "quote": q,
                            "found_in_text": re.sub(r"\s+", " ", q.lower().strip(" .,;:\"'")) in src})
    return out


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/intake", dependencies=[Depends(rate_limit)])
def intake(req: IntakeRequest, llm=Depends(get_llm)):
    t0 = time.perf_counter()
    try:
        rec = pipelines.run(llm, [req.text], req.pipeline, req.guard)[0]
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    if rec["intake"] is None:
        raise HTTPException(502, "the model did not return a valid intake; try again")
    return {"intake": rec["intake"], "read": rec["read"], "word_alerts": word_alerts(req.text),
            "evidence_checks": evidence_checks(rec["intake"], rec["read"]),
            "pipeline": req.pipeline, "guard": req.guard, "latency_ms": round((time.perf_counter() - t0) * 1000)}


@lru_cache(maxsize=1)
def get_asr():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    return WhisperModel(os.environ.get("PUENTE_ASR", "small"), device="cpu", compute_type="int8")


@api.post("/transcribe", dependencies=[Depends(rate_limit)])
async def transcribe(audio: UploadFile = File(...), asr=Depends(get_asr)):
    if asr is None:
        raise HTTPException(501, "speech input needs faster-whisper (pip install faster-whisper)")
    data = await audio.read()
    if len(data) > 10 * 2**20:
        raise HTTPException(413, "audio over 10 MB")
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or "a.webm").suffix) as f:
        f.write(data)
        f.flush()
        segments, info = asr.transcribe(f.name, language="es", vad_filter=True)
        text = " ".join(s.text.strip() for s in segments)
    return {"text": text, "duration_s": round(info.duration, 1)}


@api.get("/capabilities")
def capabilities():
    """What this deployment can do, so the page can hide speech input where it is not installed (e.g. Vercel)."""
    import importlib.util
    return {"backend": os.environ.get("PUENTE_BACKEND", "gemini"),
            "speech": importlib.util.find_spec("faster_whisper") is not None,
            "rate_per_min": RATE}


@app.get("/{name}.json")
def public_json(name: str):
    """results.json and example.json for the page (Vercel serves public/ directly; this covers local and Docker)."""
    if name not in ("results", "example"):
        raise HTTPException(404)
    return FileResponse(PUBLIC / f"{name}.json")


app.include_router(api)                  # local / Docker: /intake, /health ...
app.include_router(api, prefix="/api")   # same routes under /api, which is what the page calls (and Vercel serves)


@app.get("/")
def page():
    return FileResponse(PUBLIC / "index.html")
