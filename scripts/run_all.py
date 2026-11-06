"""Review prior-auth request files from the command line.

    python scripts/run_all.py path/to/requests/ [more files or folders] [--out results/]
                              [--expected expected.json] [--review-date YYYY-MM-DD] [--fresh]

For each request this writes <REQ>.json (full packet + agent trace) and
<REQ>_letter.txt, plus summary.md and evaluation.json for the batch.
``--expected`` optionally scores the run against {request_id: {"recommendation": ...}}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="request files (.md/.txt) or folders of them")
    ap.add_argument("--out", type=Path, default=Path("outputs"), help="results folder (default: outputs/)")
    ap.add_argument("--expected", type=Path, help="optional JSON of expected recommendations to score against")
    ap.add_argument("--review-date", default=os.getenv("MERIDIAN_REVIEW_DATE", date.today().isoformat()))
    ap.add_argument("--fresh", action="store_true", help="clear reviewer memory/checkpoints before running")
    args = ap.parse_args()
    os.environ["MERIDIAN_REVIEW_DATE"] = args.review_date

    from meridian_pa import config
    from meridian_pa.data_loader import load_gold_labels, load_requests
    from meridian_pa.evaluation import evaluate_run
    from meridian_pa.graph import Reviewer
    from meridian_pa.report import packet_markdown

    if args.fresh and config.STATE_DIR.exists():
        shutil.rmtree(config.STATE_DIR)
    config.ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)

    requests = load_requests(args.inputs)
    if not requests:
        print("No request files found.", file=sys.stderr)
        return 2
    reviewer = Reviewer()
    states = {}
    for req in requests:
        st = reviewer.review(req)
        states[req.request_id] = st
        p = st["packet"]
        (args.out / f"{req.request_id}.json").write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")
        (args.out / f"{req.request_id}_letter.txt").write_text(p["letter"], encoding="utf-8")
        print(f"{req.request_id}: {p['recommendation']:<8} basis={p['primary_basis']:<12} "
              f"confidence={p['confidence']:<6} flags={len(p['flags'])} guardrails_ok={st['output_validation']['ok']}")

    expected = load_gold_labels(args.expected) if args.expected else {}
    report = evaluate_run(states, expected)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Prior Authorization Review Run", "", f"Review date: {args.review_date}", ""]
    md += [packet_markdown(st) for st in states.values()]
    md += ["## Evaluation", "", "```json", json.dumps(report["aggregate"], indent=2), "```"]
    (args.out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Results written to {args.out.resolve()}")
    if expected and report["aggregate"]["decision_accuracy"] != 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
