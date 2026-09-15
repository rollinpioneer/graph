"""Independent-state holdout CLI. Generator path must not import scorer."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def cmd_scan(a):
    from .history_scan import scan
    result = scan(Path(a.repo))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"collision": result["collision"], "hits": len(result["hits"])}))
    return 2 if result["collision"] else 0

def cmd_generate(a):
    # Imported here only. This module does not import reward.
    from .history_scan import scan
    from .generator import generate_all
    scanned = scan(Path(a.repo))
    if scanned["collision"]:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out)/"history_scan.json").write_text(json.dumps(scanned, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"collision": True, "hits": scanned["hits"][:10]}))
        return 2
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/"history_scan.json").write_text(json.dumps(scanned, indent=2)+"\n", encoding="utf-8")
    stats = generate_all(Path(a.out))
    print(json.dumps(stats))
    return 0

def cmd_score(a):
    from .scorer import score_holdout
    stats = score_holdout(Path(a.repo), Path(a.holdout), Path(a.out))
    print(json.dumps(stats))
    return 0

def cmd_evaluate(a):
    from .evaluate import evaluate
    decision = evaluate(Path(a.holdout), Path(a.scored), Path(a.out))
    print(json.dumps({"independent_state_holdout_passed": decision["independent_state_holdout_passed"],
                      "gates": decision["gates"]}))
    return 0 if decision["independent_state_holdout_passed"] else 2

def main(argv=None):
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("scan"); q.add_argument("--repo", required=True); q.add_argument("--out", required=True)
    q = s.add_parser("generate"); q.add_argument("--repo", required=True); q.add_argument("--out", required=True)
    q = s.add_parser("score"); q.add_argument("--repo", required=True); q.add_argument("--holdout", required=True); q.add_argument("--out", required=True)
    q = s.add_parser("evaluate"); q.add_argument("--holdout", required=True); q.add_argument("--scored", required=True); q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"scan": cmd_scan, "generate": cmd_generate, "score": cmd_score, "evaluate": cmd_evaluate}[a.cmd](a)

if __name__ == "__main__":
    raise SystemExit(main())