"""The structured intake every pipeline must return, in English, with closed vocabularies so scoring is exact.

A symptom is `present` or `denied`; a symptom the patient never mentions is simply absent (never "denied").
Every list item carries `evidence`: a verbatim quote from the text the pipeline read, so invented fields are
detectable.
"""
from typing import Literal

from pydantic import BaseModel, Field

SYMPTOMS = [
    "fever", "cough", "sore_throat", "nasal_congestion", "headache", "nausea", "vomiting", "diarrhea", "constipation",
    "abdominal_pain", "epigastric_pain", "mouth_pain", "chest_pain", "shortness_of_breath", "dizziness", "sweating",
    "fatigue", "drowsiness", "rash", "vaginal_bleeding", "anxiety", "food_poisoning", "alcohol_intoxication",
    "drug_overdose", "low_blood_sugar",
]
CONDITIONS = ["diabetes", "hypertension", "asthma"]
RED_FLAGS = ["chest_pain", "shortness_of_breath", "suicidal_ideation", "bleeding_in_pregnancy", "overdose"]
FREQUENCIES = ["once_daily", "twice_daily", "three_times_daily", "every_8_hours", "as_needed"]

Symptom = Literal[tuple(SYMPTOMS)]  # type: ignore[valid-type]
Condition = Literal[tuple(CONDITIONS)]  # type: ignore[valid-type]
RedFlag = Literal[tuple(RED_FLAGS)]  # type: ignore[valid-type]
Frequency = Literal[tuple(FREQUENCIES)]  # type: ignore[valid-type]


class SymptomItem(BaseModel):
    name: Symptom
    status: Literal["present", "denied"]
    evidence: str = ""


class ConditionItem(BaseModel):
    name: Condition
    evidence: str = ""


class Medication(BaseModel):
    name: str = Field(description="generic drug name in English, lowercase, e.g. 'metformin'")
    dose: str = Field(default="", description="number and unit, e.g. '500 mg'; empty if not stated")
    frequency: Frequency | None = None
    evidence: str = ""


class Allergy(BaseModel):
    substance: str = Field(description="English, lowercase, e.g. 'penicillin'")
    evidence: str = ""


class Term(BaseModel):
    term: str = Field(description="the patient's own Spanish words")
    note: str = ""


class Intake(BaseModel):
    chief_complaint: str = ""
    symptoms: list[SymptomItem] = []
    conditions: list[ConditionItem] = []
    medications: list[Medication] = []
    allergies: list[Allergy] = []
    duration_days: int | None = Field(default=None, description="how long the main problem has lasted, in days")
    pregnant: Literal["yes", "no", "not_mentioned"] = "not_mentioned"
    red_flags: list[RedFlag] = []
    needs_clarification: list[Term] = Field(default=[], description="words whose meaning the clinician must confirm")
    culture_bound: list[Term] = Field(default=[], description="cultural idioms of distress, kept in Spanish")
    summary_en: str = ""


def vocab_text():
    """The closed vocabularies, as given to every model in its prompt."""
    return (f"symptoms.name: {', '.join(SYMPTOMS)}\n"
            f"symptoms.status: present, denied\n"
            f"conditions.name: {', '.join(CONDITIONS)}\n"
            f"red_flags: {', '.join(RED_FLAGS)}\n"
            f"medications.frequency: {', '.join(FREQUENCIES)}")
