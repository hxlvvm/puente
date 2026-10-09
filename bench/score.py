"""Deterministic scoring of predicted intakes against the template gold. No LLM judges anything here.

Headline numbers:
- pitfall pass rate: a pitfall/control/ambiguous case passes only if every required item is present and every
  forbidden item is absent (see `checks` in data/cases.jsonl);
- red-flag recall: safety, reported on its own;
- field-level micro F1 for symptoms (name + present/denied), conditions, medications (name + dose + frequency),
  allergies; pregnancy and duration accuracy;
- over-asking: clarification requests on cases that need none;
- grounding: share of evidence quotes found verbatim in the text the pipeline read.
"""
import re
import unicodedata
from collections import defaultdict

KNOWN_MEDS = ["metformin", "losartan", "ibuprofen", "amoxicillin", "paracetamol", "acetaminophen"]
FREQ_ALIAS = {"every_8_hours": "three_times_daily"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def singular(s):
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in norm(s).split())


def med_name(s):
    n = norm(s)
    return next((m for m in KNOWN_MEDS if m in n), n)


def dose(s):
    n = re.sub(r"\b(miligramos?|milligrams?)\b", "mg", norm(s))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)\b", n)
    return f"{m.group(1)} {m.group(2)}" if m else n


def allergy(s):
    n = norm(s)
    return "sulfa" if "sulf" in n else "penicillin" if "penicil" in n else n


def as_sets(d):
    """Intake dict (gold or predicted) -> comparable sets/values."""
    d = d or {}
    names = lambda xs, key="name": {x[key] if isinstance(x, dict) else x for x in xs or []}  # noqa: E731
    return {
        "symptoms": {(s["name"], s.get("status", "present")) for s in d.get("symptoms") or []},
        "conditions": names(d.get("conditions")),
        "medications": {(med_name(m.get("name")), dose(m.get("dose", "")),
                         FREQ_ALIAS.get(m.get("frequency"), m.get("frequency"))) for m in d.get("medications") or []},
        "medication_names": {med_name(m.get("name")) for m in d.get("medications") or []},
        "allergies": {allergy(a["substance"] if isinstance(a, dict) else a) for a in d.get("allergies") or []},
        "red_flags": set(d.get("red_flags") or []),
        "clarify": {singular(t["term"] if isinstance(t, dict) else t) for t in d.get("needs_clarification") or []},
        "culture": {singular(t["term"] if isinstance(t, dict) else t) for t in d.get("culture_bound") or []},
        "pregnant": d.get("pregnant") or "not_mentioned",
        "duration_days": d.get("duration_days"),
    }


def has_term(terms, wanted):
    w = singular(wanted)
    return any(w in t for t in terms)


def check(pred, checks):
    """True if the prediction satisfies every pitfall check of the case."""
    present = {n for n, st in pred["symptoms"] if st == "present"}
    ok = all(s in present for s in checks.get("require_symptoms", []))
    ok &= not any(s in present for s in checks.get("forbid_symptoms", []))
    ok &= all(c in pred["conditions"] for c in checks.get("require_conditions", []))
    ok &= not any(c in pred["conditions"] for c in checks.get("forbid_conditions", []))
    ok &= all(r in pred["red_flags"] for r in checks.get("require_red_flags", []))
    ok &= all(has_term(pred["clarify"], t) for t in checks.get("require_clarify", []))
    ok &= all(has_term(pred["culture"], t) for t in checks.get("require_culture", []))
    if checks.get("forbid_culture"):
        ok &= not pred["culture"]
    if "pregnant" in checks:
        ok &= pred["pregnant"] == checks["pregnant"]
    if "pregnant_not" in checks:
        ok &= pred["pregnant"] != checks["pregnant_not"]
    return bool(ok)


def grounding(pred_raw, source_text):
    """(quotes found verbatim in the source, quotes given). Whitespace and case are ignored."""
    src = re.sub(r"\s+", " ", source_text.lower())
    quotes = [x.get("evidence", "") for key in ("symptoms", "conditions", "medications", "allergies")
              for x in (pred_raw or {}).get(key) or [] if isinstance(x, dict) and x.get("evidence")]
    found = sum(re.sub(r"\s+", " ", q.lower().strip(" .,;:\"'«»")) in src for q in quotes)
    return found, len(quotes)


def score_case(case, pred_raw, source_text=None):
    g, p = as_sets(case["gold"]), as_sets(pred_raw)
    out = {"id": case["id"], "family": case["family"], "variant": case["variant"], "valid": pred_raw is not None}
    for f in ("symptoms", "conditions", "medications", "medication_names", "allergies", "red_flags"):
        out[f] = (len(g[f] & p[f]), len(p[f]), len(g[f]))          # tp, predicted, gold
    out["pregnant_ok"] = g["pregnant"] == p["pregnant"]
    out["duration_ok"] = g["duration_days"] == p["duration_days"]
    out["over_ask"] = bool(p["clarify"]) and not g["clarify"]
    out["pitfall_pass"] = check(p, case["checks"]) if case["checks"] else None
    out["grounding"] = grounding(pred_raw, source_text if source_text is not None else case["text"])
    fields = ("symptoms", "conditions", "medications", "allergies", "red_flags")
    out["exact"] = all(g[f] == p[f] for f in fields) and out["pregnant_ok"] and out["duration_ok"]
    return out


def prf(tp, npred, ngold):
    p = tp / npred if npred else 1.0
    r = tp / ngold if ngold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarize(rows):
    s = {"n": len(rows), "valid_rate": sum(r["valid"] for r in rows) / len(rows)}
    for f in ("symptoms", "conditions", "medications", "medication_names", "allergies", "red_flags"):
        tp, npred, ngold = (sum(r[f][i] for r in rows) for i in range(3))
        s[f] = dict(zip(("precision", "recall", "f1"), prf(tp, npred, ngold)))
    s["red_flag_recall"] = s["red_flags"]["recall"]
    s["pregnant_acc"] = sum(r["pregnant_ok"] for r in rows) / len(rows)
    s["duration_acc"] = sum(r["duration_ok"] for r in rows) / len(rows)
    s["exact_case_acc"] = sum(r["exact"] for r in rows) / len(rows)
    no_ask = [r for r in rows if not (r["family"] == "intoxicado" and r["variant"] == "ambiguous")]
    s["over_ask_rate"] = sum(r["over_ask"] for r in no_ask) / len(no_ask)
    found, total = (sum(r["grounding"][i] for r in rows) for i in range(2))
    s["grounding"] = found / total if total else None
    pit = [r for r in rows if r["pitfall_pass"] is not None]
    s["pitfall_pass"] = sum(r["pitfall_pass"] for r in pit) / len(pit) if pit else None
    by = defaultdict(list)
    for r in pit:
        by[(r["family"], r["variant"])].append(r["pitfall_pass"])
    s["pitfall_by_variant"] = {f"{f}/{v}": sum(x) / len(x) for (f, v), x in sorted(by.items())}
    fam = defaultdict(list)
    for r in pit:
        fam[r["family"]].append(r["pitfall_pass"])
    s["pitfall_by_family"] = {f: sum(x) / len(x) for f, x in sorted(fam.items())}
    return s
