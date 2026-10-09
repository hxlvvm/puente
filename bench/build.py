"""Build the benchmark: synthetic Spanish patient messages whose gold intake comes from the templates themselves.

Every pitfall comes as a minimal set: the misleading word in its real sense, a control where the literal reading
is right, and (where the meaning truly depends on context) an ambiguous case whose correct answer is to ask.
  python -m bench.build            -> data/cases.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELD_OUT = {"constipado", "boca_estomago"}       # not in the pitfall guard's lexicon (see puente/lexicon.py)


def S(name, status="present"):
    return {"name": name, "status": status}


# (family, variant, spanish template, gold, checks, first_person)
# {o} -> o/a by the patient's sex. gold keys: symptoms, conditions, medications, red_flags, pregnant, duration_days,
# needs_clarification, culture_bound. checks: require_*/forbid_* used by the pitfall score.
MAIN = [
    # --- intoxicado: poisoned / sick from something taken in -------------------------------------------------
    ("intoxicado", "food", "Comí mariscos en una boda y desde anoche estoy intoxicad{o}, con vómito y diarrea.",
     {"symptoms": [S("food_poisoning"), S("vomiting"), S("diarrhea")], "duration_days": 1},
     {"require_symptoms": ["food_poisoning"], "forbid_symptoms": ["alcohol_intoxication", "drug_overdose"]}, True),
    ("intoxicado", "food", "Creo que algo que comí en la calle me cayó mal; estoy intoxicad{o} y tengo náuseas.",
     {"symptoms": [S("food_poisoning"), S("nausea")]},
     {"require_symptoms": ["food_poisoning"], "forbid_symptoms": ["alcohol_intoxication", "drug_overdose"]}, True),
    ("intoxicado", "food", "Ayer cené pollo recalentado y quedé intoxicad{o}: vomité toda la noche.",
     {"symptoms": [S("food_poisoning"), S("vomiting")], "duration_days": 1},
     {"require_symptoms": ["food_poisoning"], "forbid_symptoms": ["alcohol_intoxication", "drug_overdose"]}, True),
    ("intoxicado", "alcohol", "Anoche tomé demasiado tequila y terminé intoxicad{o} con alcohol; hoy no paro de vomitar.",
     {"symptoms": [S("alcohol_intoxication"), S("vomiting")], "duration_days": 1},
     {"require_symptoms": ["alcohol_intoxication"], "forbid_symptoms": ["food_poisoning"]}, True),
    ("intoxicado", "alcohol", "Bebí muchas cervezas en la fiesta y me intoxiqué con el alcohol; ahora me duele la cabeza.",
     {"symptoms": [S("alcohol_intoxication"), S("headache")]},
     {"require_symptoms": ["alcohol_intoxication"], "forbid_symptoms": ["food_poisoning"]}, True),
    ("intoxicado", "drug", "Mi hijo se tomó por accidente las pastillas para dormir de su abuela y está intoxicado, muy somnoliento.",
     {"symptoms": [S("drug_overdose"), S("drowsiness")], "red_flags": ["overdose"]},
     {"require_red_flags": ["overdose"], "forbid_symptoms": ["food_poisoning", "alcohol_intoxication"]}, False),
    ("intoxicado", "drug", "Mi hija se tragó varias pastillas de paracetamol por accidente y está intoxicada.",
     {"symptoms": [S("drug_overdose")], "red_flags": ["overdose"]},
     {"require_red_flags": ["overdose"], "forbid_symptoms": ["food_poisoning", "alcohol_intoxication"]}, False),
    ("intoxicado", "ambiguous", "Mi hijo está intoxicado y muy dormido, no sé qué le pasó.",
     {"symptoms": [S("drowsiness")], "needs_clarification": ["intoxicado"]},
     {"require_clarify": ["intoxicado"],
      "forbid_symptoms": ["alcohol_intoxication", "drug_overdose", "food_poisoning"]}, False),
    ("intoxicado", "ambiguous", "Mi esposo llegó a la casa intoxicado y ahora está muy somnoliento.",
     {"symptoms": [S("drowsiness")], "needs_clarification": ["intoxicado"]},
     {"require_clarify": ["intoxicado"],
      "forbid_symptoms": ["alcohol_intoxication", "drug_overdose", "food_poisoning"]}, False),
    # --- constipado: has a cold (held out of the guard) ------------------------------------------------------
    ("constipado", "pitfall", "Estoy constipad{o}: tengo la nariz tapada, mocos y estornudos.",
     {"symptoms": [S("nasal_congestion")]},
     {"require_symptoms": ["nasal_congestion"], "forbid_symptoms": ["constipation"]}, True),
    ("constipado", "pitfall", "Llevo tres días constipad{o}, con la nariz congestionada y un poco de tos.",
     {"symptoms": [S("nasal_congestion"), S("cough")], "duration_days": 3},
     {"require_symptoms": ["nasal_congestion"], "forbid_symptoms": ["constipation"]}, True),
    ("constipado", "control", "Estoy estreñid{o}; no he podido ir al baño en cuatro días.",
     {"symptoms": [S("constipation")], "duration_days": 4},
     {"require_symptoms": ["constipation"], "forbid_symptoms": ["nasal_congestion"]}, True),
    ("constipado", "control", "Tengo estreñimiento desde hace una semana.",
     {"symptoms": [S("constipation")], "duration_days": 7},
     {"require_symptoms": ["constipation"], "forbid_symptoms": ["nasal_congestion"]}, True),
    # --- embarazada: pregnant (not embarrassed) --------------------------------------------------------------
    ("embarazada", "pitfall", "Estoy embarazada de ocho semanas y tengo sangrado vaginal desde esta mañana.",
     {"symptoms": [S("vaginal_bleeding")], "pregnant": "yes", "red_flags": ["bleeding_in_pregnancy"]},
     {"pregnant": "yes", "require_red_flags": ["bleeding_in_pregnancy"]}, "f"),
    ("embarazada", "pitfall", "Tengo cinco meses de embarazo y me duele mucho la cabeza.",
     {"symptoms": [S("headache")], "pregnant": "yes"},
     {"pregnant": "yes"}, "f"),
    ("embarazada", "control", "Me da vergüenza decirlo, pero tengo diarrea desde ayer.",
     {"symptoms": [S("diarrhea")], "duration_days": 1},
     {"pregnant_not": "yes"}, True),
    ("embarazada", "control", "Estoy muy avergonzad{o} por molestar, pero tengo fiebre desde hace dos días.",
     {"symptoms": [S("fever")], "duration_days": 2},
     {"pregnant_not": "yes"}, True),
    # --- azúcar: diabetes / low sugar / literal sugar ----------------------------------------------------------
    ("azucar", "diabetes", "Tengo azúcar desde hace años y tomo metformina de 850 miligramos dos veces al día.",
     {"conditions": ["diabetes"], "medications": [("metformin", "850 mg", "twice_daily")]},
     {"require_conditions": ["diabetes"]}, True),
    ("azucar", "diabetes", "Padezco del azúcar y últimamente me siento muy cansad{o}.",
     {"conditions": ["diabetes"], "symptoms": [S("fatigue")]},
     {"require_conditions": ["diabetes"]}, True),
    ("azucar", "low", "Se me bajó el azúcar en la mañana; me sentí maread{o} y sudando frío.",
     {"symptoms": [S("low_blood_sugar"), S("dizziness"), S("sweating")]},
     {"require_symptoms": ["low_blood_sugar"]}, True),
    ("azucar", "control", "Me gusta el café con mucha azúcar, pero hoy tengo dolor de cabeza.",
     {"symptoms": [S("headache")]},
     {"forbid_conditions": ["diabetes"], "forbid_symptoms": ["low_blood_sugar"]}, True),
    # --- ataque de nervios: cultural idiom of distress ---------------------------------------------------------
    ("nervios", "pitfall", "Desde que murió mi esposo me dan ataques de nervios: grito, tiemblo y siento que pierdo el control.",
     {"culture_bound": ["ataque de nervios"]},
     {"require_culture": ["ataque de nervios"]}, "f"),
    ("nervios", "pitfall", "Mi mamá tuvo un ataque de nervios después de una discusión familiar.",
     {"culture_bound": ["ataque de nervios"]},
     {"require_culture": ["ataque de nervios"]}, False),
    ("nervios", "control", "Me siento muy nervios{o} por un examen y no puedo dormir.",
     {"symptoms": [S("anxiety")]},
     {"require_symptoms": ["anxiety"], "forbid_culture": True}, True),
    # --- boca del estómago: epigastrium (held out of the guard) -----------------------------------------------
    ("boca_estomago", "pitfall", "Me duele la boca del estómago después de comer.",
     {"symptoms": [S("epigastric_pain")]},
     {"require_symptoms": ["epigastric_pain"], "forbid_symptoms": ["mouth_pain"]}, True),
    ("boca_estomago", "pitfall", "Siento ardor en la boca del estómago desde hace tres días.",
     {"symptoms": [S("epigastric_pain")], "duration_days": 3},
     {"require_symptoms": ["epigastric_pain"], "forbid_symptoms": ["mouth_pain"]}, True),
    ("boca_estomago", "control", "Me duele la boca por una muela picada.",
     {"symptoms": [S("mouth_pain")]},
     {"require_symptoms": ["mouth_pain"], "forbid_symptoms": ["epigastric_pain"]}, True),
    # --- general intake (no pitfall) -------------------------------------------------------------------------
    ("general", "fever_cough", "Tengo fiebre y tos desde hace {n} días.",
     {"symptoms": [S("fever"), S("cough")], "duration_days": "{n}"}, {}, True),
    ("general", "headache", "Me duele mucho la cabeza desde hace {n} días.",
     {"symptoms": [S("headache")], "duration_days": "{n}"}, {}, True),
    ("general", "throat", "Tengo dolor de garganta y fiebre.",
     {"symptoms": [S("sore_throat"), S("fever")]}, {}, True),
    ("general", "spanglish_rash", "Me salió un rash en los brazos hace {n} días.",
     {"symptoms": [S("rash")], "duration_days": "{n}"}, {}, True),
    ("general", "spanglish_throat", "Tengo un sore throat horrible y me siento muy cansad{o}.",
     {"symptoms": [S("sore_throat"), S("fatigue")]}, {}, True),
    ("general", "chest", "Tengo dolor de pecho y me falta el aire.",
     {"symptoms": [S("chest_pain"), S("shortness_of_breath")], "red_flags": ["chest_pain", "shortness_of_breath"]},
     {}, True),
    ("general", "chest_arm", "Siento una presión en el pecho que se me corre al brazo izquierdo.",
     {"symptoms": [S("chest_pain")], "red_flags": ["chest_pain"]}, {}, True),
    ("general", "suicidal", "Últimamente he pensado en quitarme la vida.",
     {"red_flags": ["suicidal_ideation"]}, {}, True),
    ("general", "gi", "Tengo diarrea y dolor de estómago desde ayer.",
     {"symptoms": [S("diarrhea"), S("abdominal_pain")], "duration_days": 1}, {}, True),
]

# Optional extra sentences (first-person cases only): (template, gold, symptom it would conflict with)
EXTRAS = [
    ("No tengo fiebre.", {"symptoms": [S("fever", "denied")]}, "fever"),
    ("No he tenido vómito.", {"symptoms": [S("vomiting", "denied")]}, "vomiting"),
    ("No tengo dolor de pecho.", {"symptoms": [S("chest_pain", "denied")]}, "chest_pain"),
    ("Tampoco me falta el aire.", {"symptoms": [S("shortness_of_breath", "denied")]}, "shortness_of_breath"),
    ("Tomo losartán de 50 miligramos una vez al día.", {"medications": [("losartan", "50 mg", "once_daily")]}, None),
    ("Uso ibuprofeno de 400 miligramos cuando lo necesito.", {"medications": [("ibuprofen", "400 mg", "as_needed")]}, "ibuprofen"),
    ("Tomo amoxicilina de 500 miligramos cada ocho horas.", {"medications": [("amoxicillin", "500 mg", "three_times_daily")]}, None),
    ("Soy alérgic{o} a la penicilina.", {"allergies": ["penicillin"]}, None),
    ("Tengo alergia a las sulfas.", {"allergies": ["sulfa"]}, None),
    ("Tengo presión alta.", {"conditions": ["hypertension"]}, "hypertension"),
    ("Tengo asma desde niñ{o}.", {"conditions": ["asthma"]}, "asthma"),
    ("No estoy embarazada.", {"pregnant": "no"}, "pregnant"),
]
OPENERS = ["", "Hola, doctor. ", "Buenos días. ", "Buenas tardes, quería consultarle algo. "]


def fill(template, sex, n):
    return template.replace("{o}", "o" if sex == "m" else "a").replace("{n}", str(n))


def merge(gold, add):
    for k, v in add.items():
        if isinstance(v, list):
            gold.setdefault(k, [])
            gold[k] += [x for x in v if x not in gold[k]]
        else:
            gold[k] = v
    return gold


def to_intake(gold):
    """Template gold -> the same shape as puente.schema.Intake (without evidence and free text)."""
    return {"symptoms": gold.get("symptoms", []),
            "conditions": gold.get("conditions", []),
            "medications": [{"name": a, "dose": b, "frequency": c} for a, b, c in gold.get("medications", [])],
            "allergies": gold.get("allergies", []),
            "duration_days": gold.get("duration_days"),
            "pregnant": gold.get("pregnant", "not_mentioned"),
            "red_flags": gold.get("red_flags", []),
            "needs_clarification": gold.get("needs_clarification", []),
            "culture_bound": gold.get("culture_bound", [])}


def build(seed=13, per_pitfall=6, per_general=16):
    rng = random.Random(seed)
    cases, seen = [], set()
    for family, variant, tpl, gold0, checks, first_person in MAIN:
        reps = per_general if family == "general" else per_pitfall
        for _ in range(reps * 3):
            if sum(c["template"] == tpl for c in cases) >= reps:
                break
            sex = "f" if first_person == "f" else rng.choice("mf")
            n = rng.randint(2, 6)
            text = rng.choice(OPENERS) + fill(tpl, sex, n)
            gold = json.loads(json.dumps(gold0).replace('"{n}"', str(n)))
            if first_person:
                taken = {s["name"] for s in gold.get("symptoms", [])} | set(gold.get("conditions", []))
                taken |= {m[0] for m in gold.get("medications", [])} | ({"pregnant"} if "pregnant" in gold else set())
                pool = [e for e in EXTRAS if e[2] not in taken and not (e[2] == "pregnant" and sex == "m")]
                for tpl_x, add, _ in rng.sample(pool, rng.randint(0, 2)):
                    text += " " + fill(tpl_x, sex, n)
                    merge(gold, json.loads(json.dumps(add)))
            if text in seen:
                continue
            seen.add(text)
            cases.append({"id": f"c{len(cases):04d}", "family": family, "variant": variant,
                          "held_out_from_guard": family in HELD_OUT, "template": tpl, "text": text,
                          "gold": to_intake(gold), "checks": checks})
    return cases


def main():
    cases = build()
    out = ROOT / "data/cases.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    fams = {}
    for c in cases:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
    print(len(cases), "cases", fams)


if __name__ == "__main__":
    main()
