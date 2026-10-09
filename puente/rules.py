"""Keyword baseline: a literal Spanish->English dictionary with simple negation, no model.

It translates false friends literally on purpose ('intoxicado' -> intoxicated, 'constipado' -> constipated), which
is what a naive dictionary lookup does. It is the floor the LLM pipelines are measured against, and it lets the
API run end to end without any key (PUENTE_BACKEND=rules).
"""
import json
import re

SYMPTOM_WORDS = [
    (r"fiebre", "fever"), (r"\btos\b", "cough"), (r"garganta|sore throat", "sore_throat"),
    (r"nariz tapada|congestionad|mocos", "nasal_congestion"), (r"cabeza", "headache"), (r"n[aá]usea", "nausea"),
    (r"v[oó]mit", "vomiting"), (r"diarrea", "diarrhea"), (r"estre[nñ]|constipad", "constipation"),
    (r"dolor de est[oó]mago", "abdominal_pain"), (r"\bboca\b", "mouth_pain"), (r"pecho", "chest_pain"),
    (r"falta el aire", "shortness_of_breath"), (r"mare", "dizziness"), (r"sudando|sudor", "sweating"),
    (r"cansad", "fatigue"), (r"somnolient|dormid", "drowsiness"), (r"\brash\b|sarpullido", "rash"),
    (r"sangrado", "vaginal_bleeding"), (r"nervios", "anxiety"), (r"intoxic", "alcohol_intoxication"),
]
MEDS = {"losart": "losartan", "metformin": "metformin", "ibuprofen": "ibuprofen", "amoxicilin": "amoxicillin"}
FREQ = [(r"una vez al d[ií]a", "once_daily"), (r"dos veces al d[ií]a", "twice_daily"),
        (r"tres veces al d[ií]a|cada ocho horas", "three_times_daily"), (r"cuando lo necesito", "as_needed")]
NUM = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7}


def extract(text):
    t = text.lower()
    sentences = re.split(r"[.;:]", t)
    symptoms = {}
    for s in sentences:
        negated = bool(re.search(r"\b(no|tampoco)\b", s))
        for pat, name in SYMPTOM_WORDS:
            if re.search(pat, s) and name not in symptoms:
                symptoms[name] = {"name": name, "status": "denied" if negated else "present", "evidence": s.strip()}
    meds = []
    for s in sentences:
        for key, name in MEDS.items():
            if key in s:
                dose = re.search(r"(\d+)\s*miligramos", s)
                freq = next((f for p, f in FREQ if re.search(p, s)), None)
                meds.append({"name": name, "dose": f"{dose.group(1)} mg" if dose else "", "frequency": freq,
                             "evidence": s.strip()})
    allergies = [{"substance": "penicillin" if "penicil" in s else "sulfa", "evidence": s.strip()}
                 for s in sentences if re.search(r"al[eé]rg", s) and ("penicil" in s or "sulfa" in s)]
    conditions = [{"name": n, "evidence": s.strip()} for s in sentences
                  for p, n in ((r"presi[oó]n alta", "hypertension"), (r"\basma\b", "asthma")) if re.search(p, s)]
    dur = None
    m = re.search(r"desde hace (\d+|un[a]?|dos|tres|cuatro|cinco|seis|siete) (d[ií]as|semana)", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUM.get(m.group(1), 1)
        dur = n * (7 if m.group(2).startswith("semana") else 1)
    elif re.search(r"desde (ayer|anoche)|anoche|ayer", t):
        dur = 1
    pregnant = "no" if re.search(r"no estoy embarazada", t) else "yes" if re.search(r"embaraz", t) else "not_mentioned"
    present = {k for k, v in symptoms.items() if v["status"] == "present"}
    flags = [f for f, ok in (("chest_pain", "chest_pain" in present),
                             ("shortness_of_breath", "shortness_of_breath" in present),
                             ("suicidal_ideation", bool(re.search(r"quitarme la vida|suicid", t))),
                             ("bleeding_in_pregnancy", pregnant == "yes" and "vaginal_bleeding" in present)) if ok]
    return {"chief_complaint": "", "symptoms": list(symptoms.values()), "conditions": conditions,
            "medications": meds, "allergies": allergies, "duration_days": dur, "pregnant": pregnant,
            "red_flags": flags, "needs_clarification": [], "culture_bound": [], "summary_en": ""}


class Rules:
    """Looks like an LLM backend to the pipelines: returns the keyword extraction as JSON."""
    name = "keyword-rules"

    def generate(self, prompts, json_mode=True):
        out = []
        for p in prompts:
            if not json_mode:          # 'translation' step: the baseline does not translate
                out.append(p.rsplit('"""', 2)[-2] if p.count('"""') >= 2 else p)
                continue
            msg = p.rsplit('"""', 2)[-2] if p.count('"""') >= 2 else p
            out.append(json.dumps(extract(msg), ensure_ascii=False))
        return out
