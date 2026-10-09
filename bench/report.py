"""Markdown results tables from runs/*.json (full-benchmark runs only).

  python -m bench.report
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ["intoxicado", "constipado", "embarazada", "azucar", "nervios", "boca_estomago"]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def load():
    runs = []
    for p in sorted((ROOT / "runs").glob("*.json")):
        s = json.loads(p.read_text())
        if "config" in s and s["config"]["n"] >= 290:
            runs.append(s)
    order = {"keyword-rules": 0, "Qwen2.5-7B-Instruct": 1}
    return sorted(runs, key=lambda s: (order.get(s["config"]["model"], 2), s["config"]["model"],
                                      s["config"]["pipeline"] != "direct", s["config"]["guard"]))


def label(c):
    if c["model"] == "keyword-rules":
        return "Keyword dictionary (no model)"
    pipe = "read Spanish directly" if c["pipeline"] == "direct" else "translate, then extract"
    return f"{c['model']}, {pipe}{' + guard' if c['guard'] else ''}"


def write_json(runs, path):
    """Compact results for the web page's Benchmark tab."""
    out = [{"system": label(s["config"]), "model": s["config"]["model"], "pipeline": s["config"]["pipeline"],
            "guard": s["config"]["guard"], "pitfall_pass": s["pitfall_pass"],
            "pitfall_in_guard_lexicon": s.get("pitfall_in_guard_lexicon"), "pitfall_held_out": s.get("pitfall_held_out"),
            "red_flag_recall": s["red_flag_recall"], "symptom_f1": s["symptoms"]["f1"],
            "medication_f1": s["medications"]["f1"], "exact_case_acc": s["exact_case_acc"],
            "over_ask_rate": s["over_ask_rate"], "grounding": s["grounding"], "by_family": s["pitfall_by_family"],
            "ambiguous_asked": s["pitfall_by_variant"].get("intoxicado/ambiguous")} for s in runs]
    Path(path).write_text(json.dumps({"n_cases": runs[0]["n"] if runs else 0, "systems": out}, indent=1))


def main():
    import sys
    runs = load()
    if "--json" in sys.argv:
        write_json(runs, ROOT / "public/results.json")
        print("wrote public/results.json")
        return
    n = runs[0]["n"] if runs else 0
    print(f"### All {n} cases, %\n")
    print("| System | Pitfall pass | ...words in guard list | ...held-out words | Red-flag recall | Symptom F1 "
          "| Medication F1 | Whole intake exact | Asks when not needed | Quotes found in text |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in runs:
        print(f"| {label(s['config'])} | **{pct(s['pitfall_pass'])}** | {pct(s.get('pitfall_in_guard_lexicon'))} | "
              f"{pct(s.get('pitfall_held_out'))} | {pct(s['red_flag_recall'])} | {pct(s['symptoms']['f1'])} | "
              f"{pct(s['medications']['f1'])} | {pct(s['exact_case_acc'])} | {pct(s['over_ask_rate'])} | "
              f"{pct(s['grounding'])} |")
    print("\n### Pitfall pass rate by word family (pitfall, control and ambiguous cases together), %\n")
    print("| System | " + " | ".join(FAMILIES) + " |")
    print("|---|" + "---|" * len(FAMILIES))
    for s in runs:
        print(f"| {label(s['config'])} | " + " | ".join(pct(s["pitfall_by_family"].get(f)) for f in FAMILIES) + " |")
    print("\n### The ambiguous 'intoxicado' cases (correct answer: ask what was taken), %\n")
    print("| System | Asked | Food/literal sense right |")
    print("|---|---|---|")
    for s in runs:
        v = s["pitfall_by_variant"]
        print(f"| {label(s['config'])} | {pct(v.get('intoxicado/ambiguous'))} | {pct(v.get('intoxicado/food'))} |")


if __name__ == "__main__":
    main()
