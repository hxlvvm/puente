"""Two ways to turn a Spanish patient message into an English intake, each with or without the pitfall guard.

- direct:    the model reads the Spanish and fills the English form in one step;
- translate: the model first translates to English, then fills the form from the translation
             (the usual "translate, then run the English system" design).
The guard adds dictionary notes for clinical-Spanish pitfalls found in the message (puente/lexicon.py) to
whichever step reads the Spanish.
"""
import json

from pydantic import ValidationError

from puente.lexicon import guard_notes
from puente.schema import (CONDITIONS, FREQUENCIES, RED_FLAGS, SYMPTOMS, Allergy, ConditionItem, Intake, Medication,
                           SymptomItem, Term, vocab_text)

RULES = """Rules:
- Output English, except `evidence` and `term`, which must be copied word for word from the message.
- Use only the listed vocabulary values. A symptom the patient says they do NOT have is "denied"; a symptom never
  mentioned is left out (never "denied").
- duration_days: how long the main problem has lasted (yesterday or last night = 1, a week = 7); null if not said.
- pregnant: "yes" or "no" only if the message says so; otherwise "not_mentioned".
- red_flags: list every listed emergency sign the message contains.
- needs_clarification: only words whose clinical meaning is genuinely unclear from the message.
- culture_bound: cultural idioms of distress, kept in the patient's words, not turned into a diagnosis.
- Return one JSON object with exactly these keys: chief_complaint, symptoms, conditions, medications, allergies,
  duration_days, pregnant, red_flags, needs_clarification, culture_bound, summary_en."""

SHAPE = ('{"chief_complaint": str, "symptoms": [{"name", "status", "evidence"}], "conditions": [{"name", "evidence"}], '
         '"medications": [{"name", "dose", "frequency", "evidence"}], "allergies": [{"substance", "evidence"}], '
         '"duration_days": int|null, "pregnant": str, "red_flags": [str], "needs_clarification": [{"term", "note"}], '
         '"culture_bound": [{"term", "note"}], "summary_en": str}')


def notes_block(text):
    found = guard_notes(text)
    if not found:
        return ""
    return "\nNotes on Spanish words in this message (from a medical interpreting glossary):\n" + "\n".join(
        f"- {e['note']}" for e in found) + "\n"


def extract_prompt(message, language, notes=""):
    return (f"You are a clinical intake assistant for a US clinic. Read this patient message ({language}) and fill "
            f"the intake form.\n\nVocabulary:\n{vocab_text()}\n\nJSON shape: {SHAPE}\n\n{RULES}\n{notes}\n"
            f"Patient message:\n\"\"\"{message}\"\"\"")


def translate_prompt(message, notes=""):
    return ("Translate this patient message from Spanish into English for a US clinician. Translate faithfully; do "
            f"not add or drop information. Output only the translation.\n{notes}\nMessage:\n\"\"\"{message}\"\"\"")


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder(strict=False).raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def coerce(raw):
    """Model JSON -> a valid Intake dict, dropping items outside the vocabularies. Returns (intake, n_dropped)."""
    if raw is None:
        return None, 0
    dropped = 0

    def keep(items, model):
        nonlocal dropped
        good = []
        for x in items if isinstance(items, list) else []:
            try:
                good.append(model.model_validate(x if isinstance(x, dict) else {"term": x}).model_dump())
            except ValidationError:
                dropped += 1
        return good

    out = {"chief_complaint": str(raw.get("chief_complaint") or ""), "summary_en": str(raw.get("summary_en") or ""),
           "symptoms": keep(raw.get("symptoms"), SymptomItem), "conditions": keep(raw.get("conditions"), ConditionItem),
           "medications": keep([{**m, "frequency": m.get("frequency") if m.get("frequency") in FREQUENCIES else None,
                                  "dose": str(m.get("dose") or "")} if isinstance(m, dict) else m
                                 for m in (raw.get("medications") if isinstance(raw.get("medications"), list) else [])],
                                Medication),
           "allergies": keep(raw.get("allergies"), Allergy),
           "needs_clarification": keep(raw.get("needs_clarification"), Term),
           "culture_bound": keep(raw.get("culture_bound"), Term)}
    flags = [f for f in raw.get("red_flags") or [] if f in RED_FLAGS]
    dropped += len(raw.get("red_flags") or []) - len(flags)
    out["red_flags"] = flags
    d = raw.get("duration_days")
    out["duration_days"] = int(d) if isinstance(d, (int, float)) or (isinstance(d, str) and d.isdigit()) else None
    out["pregnant"] = raw.get("pregnant") if raw.get("pregnant") in ("yes", "no", "not_mentioned") else "not_mentioned"
    return Intake.model_validate(out).model_dump(), dropped


def run(llm, messages, pipeline="direct", guard=False):
    """Returns one record per message: the text the extractor read, the raw output, the coerced intake."""
    if pipeline == "direct":
        read = list(messages)
        prompts = [extract_prompt(m, "Spanish", notes_block(m) if guard else "") for m in messages]
    elif pipeline == "translate":
        read = [t.strip() for t in llm.generate([translate_prompt(m, notes_block(m) if guard else "")
                                                 for m in messages], json_mode=False)]
        prompts = [extract_prompt(t, "English, translated from Spanish") for t in read]
    else:
        raise ValueError(pipeline)
    raws = llm.generate(prompts, json_mode=True)
    # one retry for output that is not valid JSON, as a production system would do
    bad = [i for i, raw in enumerate(raws) if parse_json(raw) is None]
    if bad:
        again = llm.generate([prompts[i] + "\n\nYour previous answer was not valid JSON. Return only one valid JSON "
                              "object." for i in bad], json_mode=True)
        for i, raw in zip(bad, again):
            raws[i] = raw
    recs = []
    for i, (m, r, raw) in enumerate(zip(messages, read, raws)):
        intake, dropped = coerce(parse_json(raw))
        recs.append({"message": m, "read": r, "raw": raw, "intake": intake, "dropped": dropped, "retried": i in bad})
    return recs


__all__ = ["run", "coerce", "parse_json", "extract_prompt", "translate_prompt", "SYMPTOMS", "CONDITIONS"]
