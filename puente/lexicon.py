"""Clinical-Spanish pitfalls, each with the sources it rests on.

`in_guard` marks the entries the pitfall guard may use. Two families are deliberately left out of the guard
(constipado, boca del estómago) so the benchmark can show whether the guard helps beyond the words it was given.
"""
import re

SOURCES = {
    "cchi": "OpenExamPrep, CCHI Certified Healthcare Interpreter study guide, 'Abbreviations, Eponyms, False Cognates & "
            "Terminology Management' (open-exam-prep.com/study-guides/cchi-chi/terminology-word-structure/"
            "abbreviations-eponyms-false-cognates-terminology-management)",
    "mitre": "J. DeCamp, 'When Intoxicado ≠ Intoxicated: Avoiding False Friends and Critical Consequences in Machine "
             "Translation', MITRE, 2017 (kde.mitre.org/?p=3911)",
    "altalang": "N. Tavarez, 'The Danger of False Cognates in Healthcare', ALTA Language Services "
                "(altalang.com/beyond-words/false-cognates-healthcare/)",
    "jgim": "J Gen Intern Med 2024, doi:10.1007/s11606-024-08619-8 (case in which 'intoxicado' led to an overdose "
            "work-up)",
    "azucar": "Common Ground International, 'How to Discuss Signs and Symptoms of Diabetes in Spanish' "
              "(commongroundinternational.com/medical-spanish-blog/discuss-signs-symptoms-diabetes-spanish/); "
              "Inklingo, 'How to say I am diabetic in Spanish' (inklingo.app)",
    "rae": "Real Academia Española, Diccionario de la lengua española, entry 'estómago' (dle.rae.es/estómago): set "
           "phrase 'boca del estómago'",
    "nervios": "López-Cepero et al., Soc Psychiatry Psychiatr Epidemiol 2023, doi:10.1007/s00127-023-02601-1; "
               "Lerman Ginzburg et al., BMC Psychol 2021, doi:10.1186/s40359-021-00544-3; Park & Kim, Adv Exp Med "
               "Biol 2020, doi:10.1007/978-981-32-9705-0_12 (DSM-5 cultural concepts of distress)",
}

LEXICON = [
    {"family": "intoxicado", "pattern": r"\bintoxi(?:cad[oa]s?|c[oó]|qu[eé]|carse|caron|caci[oó]n)\b", "in_guard": True, "sources": ["cchi", "mitre", "jgim"],
     "note": "Spanish 'intoxicado/a' means poisoned or made sick by something taken in (food, a chemical, a "
             "medicine, alcohol). It does not by itself mean drunk or high. Decide from context: after food -> "
             "food_poisoning; after alcohol -> alcohol_intoxication; after pills or drugs -> drug_overdose. If the "
             "context does not say what was taken, list 'intoxicado' under needs_clarification."},
    {"family": "embarazada", "pattern": r"\bembarazad[oa]s?\b", "in_guard": True, "sources": ["cchi", "altalang"],
     "note": "Spanish 'embarazada' means pregnant, not embarrassed (embarrassed = 'avergonzada', 'me da "
             "vergüenza')."},
    {"family": "azucar", "pattern": r"\baz[uú]car\b", "in_guard": True, "sources": ["azucar"],
     "note": "Colloquially 'tengo azúcar' / 'padezco del azúcar' means the person has diabetes (high blood sugar). "
             "'Se me bajó el azúcar' means low blood sugar. Sugar in food or drink is just sugar."},
    {"family": "nervios", "pattern": r"\bataques? de nervios\b", "in_guard": True, "sources": ["nervios"],
     "note": "'Ataque de nervios' is a recognised Latino cultural idiom of distress. Keep the Spanish term under "
             "culture_bound for the clinician; do not translate it into a diagnosis."},
    {"family": "constipado", "pattern": r"\bconstipad[oa]s?\b", "in_guard": False, "sources": ["cchi", "altalang"],
     "note": "Spanish 'constipado/a' means having a cold (nasal congestion), not constipated (= 'estreñido/a')."},
    {"family": "boca_estomago", "pattern": r"\bboca del est[oó]mago\b", "in_guard": False, "sources": ["rae"],
     "note": "'La boca del estómago' is the pit of the stomach (epigastrium), not the mouth."},
]


def guard_notes(text, include_held_out=False):
    """Notes for every guard-eligible pitfall found in the text."""
    t = text.lower()
    return [e for e in LEXICON if (e["in_guard"] or include_held_out) and re.search(e["pattern"], t)]
