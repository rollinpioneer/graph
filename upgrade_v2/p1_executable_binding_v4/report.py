from __future__ import annotations
import json
from pathlib import Path
from .util import write_json, write_text, write_csv

def write_final(out: Path, payload: dict) -> None:
    write_json(out / "decision.json", payload["decision"])
    write_text(out / "report.md", payload["report_md"])
    write_text(out / "next_stage_plan.md", payload["next_stage"])
    write_text(out / "external_artifacts.tsv", payload["external_tsv"])
    write_json(out / "result_manifest.json", payload["result_manifest"])
    write_csv(out / "claim_to_evidence.csv", payload["claims"])