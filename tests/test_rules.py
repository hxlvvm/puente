from bench.build import build
from bench.run import main as run_main  # noqa: F401  (import check)
from bench.score import score_case, summarize
from puente import pipelines
from puente.rules import Rules, extract


def test_keyword_baseline_reads_literally():
    d = extract("Estoy constipado. No tengo fiebre. Tomo losartán de 50 miligramos una vez al día.")
    assert {(s["name"], s["status"]) for s in d["symptoms"]} == {("constipation", "present"), ("fever", "denied")}
    assert d["medications"][0] == {"name": "losartan", "dose": "50 mg", "frequency": "once_daily",
                                   "evidence": "tomo losartán de 50 miligramos una vez al día"}


def test_keyword_baseline_runs_through_the_pipeline_and_scorer():
    cases = build()[:40]
    recs = pipelines.run(Rules(), [c["text"] for c in cases], "direct")
    s = summarize([score_case(c, r["intake"], r["read"]) for c, r in zip(cases, recs)])
    assert s["valid_rate"] == 1.0 and s["grounding"] == 1.0


def test_keyword_baseline_handles_accents_in_allergies():
    d = extract("Soy alérgico a la penicilina. Tengo alergia a las sulfas.")
    assert {a["substance"] for a in d["allergies"]} == {"penicillin", "sulfa"}
