# Puente

**Spanish patient message in, structured English intake out, with every field tied to the patient's own words.**

Spanish-speaking patients are a large part of US healthcare, and their messages are often read through machine
translation or an English-first system. Some Spanish words look like English but mean something else: *intoxicado*
usually means made sick by something eaten or taken, not drunk; *constipado* means having a cold; *embarazada* means
pregnant. In a widely cited case, *intoxicado* read as "intoxicated" delayed treatment, and translation tools still
make that mistake.

Puente does two things:

1. **A benchmark** of 296 synthetic Spanish patient messages built from cited interpreter and dictionary sources.
   Every misleading word comes as a minimal set: the word in its real sense, a **control** where the literal reading
   is right, and for *intoxicado* **ambiguous** cases where the right answer is to ask. A system that always
   "corrects" the word fails the controls. Scoring is deterministic; no LLM judges anything.
2. **An app** (FastAPI + web page) that turns a message into an English intake form for clinician review. Every
   symptom, medication, allergy and condition quotes the patient's words, and the page checks each quote really
   appears in the message. Pitfall words get a reviewer alert with their sources.

> Demo with synthetic data only. Not a medical device, not for clinical use, never enter real patient information.

## Results

### All 296 cases, %

| System | Pitfall pass | ...words in guard list | ...held-out words | Red-flag recall | Symptom F1 | Medication F1 | Whole intake exact | Asks when not needed | Quotes found in text |
|---|---|---|---|---|---|---|---|---|---|
| Keyword dictionary (no model) | **35.5** | 32.7 | 42.9 | 89.7 | 85.4 | 100.0 | 51.0 | 0.0 | 100.0 |
| Qwen2.5-7B-Instruct, read Spanish directly | **54.6** | 50.9 | 64.3 | 89.7 | 63.6 | 97.4 | 43.2 | 8.0 | 85.5 |
| Qwen2.5-7B-Instruct, read Spanish directly + guard | **58.6** | 57.3 | 61.9 | 89.7 | 63.4 | 98.7 | 42.6 | 18.8 | 86.7 |
| Qwen2.5-7B-Instruct, translate, then extract | **53.9** | 54.5 | 52.4 | 85.9 | 55.9 | 99.4 | 47.0 | 9.4 | 79.3 |
| Qwen2.5-7B-Instruct, translate, then extract + guard | **55.9** | 56.4 | 54.8 | 85.9 | 54.7 | 98.7 | 47.6 | 8.7 | 76.1 |
| gemini-3.5-flash-lite, read Spanish directly | **93.4** | 90.9 | 100.0 | 100.0 | 98.1 | 93.5 | 85.1 | 9.7 | 96.5 |

### Pitfall pass rate by word family (pitfall, control and ambiguous cases together), %

| System | intoxicado | constipado | embarazada | azucar | nervios | boca_estomago |
|---|---|---|---|---|---|---|
| Keyword dictionary (no model) | 13.0 | 50.0 | 100.0 | 25.0 | 0.0 | 33.3 |
| Qwen2.5-7B-Instruct, read Spanish directly | 19.6 | 75.0 | 100.0 | 79.2 | 25.0 | 50.0 |
| Qwen2.5-7B-Instruct, read Spanish directly + guard | 17.4 | 70.8 | 100.0 | 75.0 | 81.2 | 50.0 |
| Qwen2.5-7B-Instruct, translate, then extract | 30.4 | 62.5 | 95.8 | 75.0 | 31.2 | 38.9 |
| Qwen2.5-7B-Instruct, translate, then extract + guard | 32.6 | 66.7 | 100.0 | 75.0 | 31.2 | 38.9 |
| gemini-3.5-flash-lite, read Spanish directly | 82.6 | 100.0 | 100.0 | 100.0 | 87.5 | 100.0 |

### The ambiguous 'intoxicado' cases (correct answer: ask what was taken, without guessing), %

| System | Asked without guessing a cause | Food sense right |
|---|---|---|
| Keyword dictionary (no model) | 0.0 | 0.0 |
| Qwen2.5-7B-Instruct, read Spanish directly | 0.0 | 16.7 |
| Qwen2.5-7B-Instruct, read Spanish directly + guard | 87.5 | 0.0 |
| Qwen2.5-7B-Instruct, translate, then extract | 0.0 | 33.3 |
| Qwen2.5-7B-Instruct, translate, then extract + guard | 0.0 | 33.3 |
| gemini-3.5-flash-lite, read Spanish directly | 50.0 | 100.0 |

Two pipelines are compared on every model: **read Spanish directly** (one call fills the English form) and
**translate, then extract** (the usual "translate, then use the English system" design). The **pitfall guard** adds
interpreter-glossary notes for known pitfall words to the prompt. Two word families (*constipado*, *boca del
estómago*) are deliberately kept out of the guard, so the table shows whether it helps beyond the words it was given.

**What the numbers show**

- Gemini 3.5 Flash-Lite reading Spanish directly passes 93.4 % of pitfall cases with 100 % red-flag recall. On the
  ambiguous *"llegó intoxicado"* it flags the word for clarification every time, but in half of those cases it also
  guesses a cause the message does not state (food poisoning, an overdose or alcohol).
- An open 7B model (Qwen2.5-7B-Instruct, Apache-2.0, run locally) passes 54.6 %. The guard raises that to 58.6 %,
  mostly by keeping *ataque de nervios* as a cultural idiom (25 → 81 %) and by asking about the ambiguous
  *intoxicado* (0 → 88 %). The cost is more unnecessary questions (8 → 19 %) and losing the food sense of
  *intoxicado* in some cases.
- A literal keyword dictionary passes 35.5 %: it gets four of the five control sets right and the false friends
  wrong, which is the failure the benchmark is built to expose.
- Translating first did not help the 7B model on pitfalls, and it lowered red-flag recall (89.7 → 85.9 %).
- Gemini runs with the guard and the translate pipeline are still in progress (free-tier quota) and will be added.

## How it works

| Part | File | What it does |
|---|---|---|
| Schema | `puente/schema.py` | English intake with closed vocabularies; symptoms are present or denied, never-mentioned is absent |
| Pitfall lexicon | `puente/lexicon.py` | Six word families, each with sources; two held out of the guard |
| Pipelines | `puente/pipelines.py` | direct / translate-then-extract, optional guard, one retry on invalid JSON, vocabulary checks |
| Backends | `puente/llm.py`, `puente/rules.py` | Gemini (REST), local Hugging Face model on a GPU, keyword baseline |
| Benchmark | `bench/build.py`, `bench/score.py`, `bench/run.py` | Template generator, deterministic scorer, resumable runner |
| App | `serve/app.py`, `public/index.html` | FastAPI (`/api/intake`, `/api/transcribe` with faster-whisper) and the reviewer page |

The data card, with every word family, its sources, the case counts and the limitations, is in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
PUENTE_BACKEND=rules .venv/bin/uvicorn serve.app:app      # keyword baseline, no key needed
GEMINI_API_KEY=... .venv/bin/uvicorn serve.app:app        # Gemini backend; open http://127.0.0.1:8000
```

Docker: `docker build -t puente . && docker run -p 8000:8000 -e GEMINI_API_KEY=... puente`
(add `--build-arg WITH_ASR=1` for voice input).

**Vercel:** import the repository, set `GEMINI_API_KEY` and `PUENTE_BACKEND=gemini` as environment variables, and
deploy. `public/` is served as the site and `api/index.py` runs the API. Voice input is not available there. Requests
are rate-limited per visitor (`PUENTE_RATE_PER_MIN`, default 8).

## Reproduce the benchmark

```bash
python -m bench.build                                        # regenerates data/cases.jsonl exactly
python -m bench.run --backend rules --pipeline direct
GEMINI_API_KEY=... python -m bench.run --backend gemini --pipeline direct --guard
python -m bench.run --backend hf --model Qwen/Qwen2.5-7B-Instruct --pipeline translate   # one GPU (jobs/run_hf.sbatch)
python -m bench.report                                       # the tables above
```

## Limitations

- Synthetic, template-built messages. They test specific, sourced pitfalls; they are not real patient data or a
  clinical validation.
- The Spanish templates were written with AI assistance and have not yet been reviewed by a native-speaking clinician
  or interpreter.
- Six word families only, mostly as met in US healthcare; regional usage varies.
- One run per configuration; temperature 0.

## Licence and sources

Code MIT. Word meanings follow the CCHI interpreter study guide, MITRE (DeCamp 2017), J Gen Intern Med 2024
(doi:10.1007/s11606-024-08619-8), the Real Academia Española dictionary, and published studies of *ataque de nervios*
(doi:10.1007/s00127-023-02601-1, doi:10.1186/s40359-021-00544-3). Qwen2.5-7B-Instruct is Apache-2.0.
