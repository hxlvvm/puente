import json

from fastapi.testclient import TestClient

from puente import pipelines
from serve.app import app, get_llm

GOOD = {"chief_complaint": "vomiting after seafood",
        "symptoms": [{"name": "food_poisoning", "status": "present", "evidence": "estoy intoxicado"},
                     {"name": "vomiting", "status": "present", "evidence": "con vómito"},
                     {"name": "not_a_symptom", "status": "present", "evidence": "x"}],
        "conditions": [], "medications": [{"name": "metformin", "dose": "850 mg", "frequency": "daily-ish"}],
        "allergies": [{"substance": "penicillin", "evidence": "alérgico a la penicilina"}],
        "duration_days": "1", "pregnant": "maybe", "red_flags": ["overdose", "bad_flag"],
        "needs_clarification": [], "culture_bound": ["ataque de nervios"], "summary_en": "Seafood, then vomiting."}


class FakeLLM:
    """Records prompts; answers translation prompts with text and extraction prompts with fixed JSON."""
    name = "fake"

    def __init__(self, answers=None):
        self.prompts, self.answers = [], list(answers or [])

    def generate(self, prompts, json_mode=True):
        self.prompts += prompts
        if not json_mode:
            return ["I ate seafood and since last night I have been poisoned, vomiting."] * len(prompts)
        return [self.answers.pop(0) if self.answers else "```json\n" + json.dumps(GOOD) + "\n```" for _ in prompts]


def test_coerce_drops_out_of_vocabulary_items():
    intake, dropped = pipelines.coerce(GOOD)
    assert [s["name"] for s in intake["symptoms"]] == ["food_poisoning", "vomiting"]
    assert intake["red_flags"] == ["overdose"] and dropped == 2
    assert intake["medications"][0]["frequency"] is None          # invalid frequency cleared, item kept
    assert intake["duration_days"] == 1 and intake["pregnant"] == "not_mentioned"
    assert intake["culture_bound"] == [{"term": "ataque de nervios", "note": ""}]


def test_parse_json_handles_fences_and_garbage():
    assert pipelines.parse_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert pipelines.parse_json("Sure! {\"a\": 1} done") == {"a": 1}
    assert pipelines.parse_json("no json here") is None


def test_guard_adds_notes_only_when_asked():
    llm = FakeLLM()
    msg = "Comí mariscos y estoy intoxicado."
    pipelines.run(llm, [msg], "direct", guard=False)
    pipelines.run(llm, [msg], "direct", guard=True)
    assert "interpreting glossary" not in llm.prompts[0] and "interpreting glossary" in llm.prompts[1]


def test_translate_pipeline_extracts_from_the_translation():
    llm = FakeLLM()
    rec = pipelines.run(llm, ["Comí mariscos y estoy intoxicado."], "translate")[0]
    assert rec["read"].startswith("I ate seafood") and len(llm.prompts) == 2
    assert "I ate seafood" in llm.prompts[1]


def test_invalid_json_is_retried_once():
    llm = FakeLLM(answers=["{broken"])
    rec = pipelines.run(llm, ["Tengo fiebre."], "direct")[0]
    assert rec["retried"] and rec["intake"] is not None and len(llm.prompts) == 2


app.dependency_overrides[get_llm] = FakeLLM
client = TestClient(app)


def test_api_intake_with_alerts_and_evidence_checks():
    r = client.post("/intake", json={"text": "Comí mariscos y desde anoche estoy intoxicado, con vómito. "
                                             "Soy alérgico a la penicilina."})
    assert r.status_code == 200
    d = r.json()
    assert {s["name"] for s in d["intake"]["symptoms"]} == {"food_poisoning", "vomiting"}
    assert [a["family"] for a in d["word_alerts"]] == ["intoxicado"] and d["word_alerts"][0]["sources"]
    assert all(c["found_in_text"] for c in d["evidence_checks"])


def test_api_validation_and_page():
    assert client.post("/intake", json={"text": "x"}).status_code == 422
    assert client.post("/intake", json={"text": "hola doctor", "pipeline": "other"}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "Not for clinical use" in page.text
