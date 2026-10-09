"""Run one configuration over the benchmark and score it.

  GEMINI_API_KEY=... python -m bench.run --backend gemini --pipeline direct --guard --limit 20
  python -m bench.run --backend hf --model Qwen/Qwen2.5-7B-Instruct --pipeline translate

Raw outputs go to runs/raw/<name>.jsonl (resumable); the score summary to runs/<name>.json.
"""
import argparse
import json
from pathlib import Path

from bench.score import score_case, summarize
from puente import pipelines

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemini", "hf", "rules"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--pipeline", choices=["direct", "translate"], default="direct")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N cases of a fixed shuffle (pilot runs)")
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args(argv)

    cases = [json.loads(line) for line in (ROOT / "data/cases.jsonl").open(encoding="utf-8")]
    if args.limit:
        import random
        cases = random.Random(7).sample(cases, args.limit)
    default = {"rules": "keyword-rules", "gemini": "gemini-3.5-flash-lite", "hf": "Qwen/Qwen2.5-7B-Instruct"}
    model_id = args.model or default[args.backend]
    model_name = model_id.split("/")[-1]
    name = f"{model_name}__{args.pipeline}{'__guard' if args.guard else ''}{f'__n{args.limit}' if args.limit else ''}"
    raw_path = ROOT / "runs/raw" / f"{name}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw_path.exists():
        for line in raw_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    print(f"{name}: {len(done)} cached, {len(todo)} to run", flush=True)
    if todo:                                   # load a model only when something is left to run
        if args.backend == "rules":
            from puente.rules import Rules
            llm = Rules()
        elif args.backend == "gemini":
            from puente.llm import Gemini
            llm = Gemini(model_id)
        else:
            from puente.llm import LocalHF
            llm = LocalHF(model_id)
    with raw_path.open("a", encoding="utf-8") as f:
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s:s + args.chunk]
            for c, rec in zip(chunk, pipelines.run(llm, [c["text"] for c in chunk], args.pipeline, args.guard)):
                rec["id"] = c["id"]
                done[c["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(s + args.chunk, len(todo))}/{len(todo)}", flush=True)

    rows = [score_case(c, done[c["id"]]["intake"], done[c["id"]]["read"]) for c in cases]
    summary = summarize(rows)
    summary.update({"config": {"model": model_name, "pipeline": args.pipeline, "guard": args.guard, "n": len(cases)},
                    "dropped_items": sum(done[c["id"]]["dropped"] for c in cases)})
    held = [r for r, c in zip(rows, cases) if c["held_out_from_guard"]]
    known = [r for r, c in zip(rows, cases) if c["checks"] and not c["held_out_from_guard"] and c["family"] != "general"]
    for key, sub in (("pitfall_in_guard_lexicon", known), ("pitfall_held_out", held)):
        vals = [r["pitfall_pass"] for r in sub if r["pitfall_pass"] is not None]
        summary[key] = sum(vals) / len(vals) if vals else None
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("pitfall_pass", "pitfall_in_guard_lexicon", "pitfall_held_out",
                                              "red_flag_recall", "exact_case_acc", "over_ask_rate", "grounding",
                                              "valid_rate", "dropped_items")}, indent=1))
    print(json.dumps(summary["pitfall_by_variant"], indent=1))


if __name__ == "__main__":
    main()
