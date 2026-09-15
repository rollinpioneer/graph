from __future__ import annotations
import json
from pathlib import Path
from .util import require_new, write_json, write_csv, write_text, sha256_file

def write_final(art: Path, out: Path):
    art=Path(art); out=require_new(out)
    parent=json.loads((art/"00_static_v1"/"parent_validation.json").read_text()) if (art/"00_static_v1"/"parent_validation.json").is_file() else {}
    forensic=json.loads((art/"01_v6_forensics_v1"/"summary.json").read_text()) if (art/"01_v6_forensics_v1"/"summary.json").is_file() else {}
    sel=json.loads((art/"03_selection_v1"/"selected_controller.json").read_text()) if (art/"03_selection_v1"/"selected_controller.json").is_file() else {}
    hold=json.loads((art/"04_engineering_holdout_v1"/"decision.json").read_text()) if (art/"04_engineering_holdout_v1"/"decision.json").is_file() else {}
    status=hold.get("status") or "RETURN_CONTROLLER_ENGINEERING_NOT_READY"
    decision={
        "engineering_status": status,
        "confirmation_passed": False,
        "policy_gain_claimed": False,
        "visual_grounding_claimed": False,
        "real_robot_claimed": False,
        "selected_controller": (sel.get("selected") or {}).get("controller_id"),
        "parent": parent,
        "forensics": forensic,
        "holdout": hold,
        "V6_reward_changed": False,
    }
    write_json(out/"decision.json", decision)
    write_csv(out/"claim_to_evidence.csv", [
        {"claim": "EEF-only return is insufficient", "evidence": str(art/"01_v6_forensics_v1/summary.json"), "status": True},
        {"claim": "return controller ready for independent confirmation", "evidence": str(art/"04_engineering_holdout_v1/decision.json"), "status": status},
        {"claim": "confirmation_passed", "evidence": "always false this stage", "status": False},
    ])
    write_json(out/"next_stage_handoff.json", {
        "frozen_controller_id": decision["selected_controller"],
        "confirmation_data_reuse_allowed": False,
        "V6_CAP_POTENTIAL_frozen": True,
        "confirmation_passed": False,
        "future_confirmation_must_use_new_families": True,
    })
    files={str(p.relative_to(art)): sha256_file(p) for p in art.rglob("*") if p.is_file() and p.stat().st_size<2_000_000}
    write_json(out/"result_manifest.json", {"n": len(files), "confirmation_passed": False, "status": status})
    write_text(out/"report.md", f"""# V6RC1 return controller engineering

Status: {status}
confirmation_passed: false
Selected: {decision['selected_controller']}
V6 reward changed: no

Forensics: {json.dumps(forensic, ensure_ascii=False)}
Holdout: {json.dumps(hold, ensure_ascii=False)[:2000]}

Limitations: TRANSLATION_COMPENSATED_FIXED_ORIENTATION; constraint-assisted MuJoCo; not a reward confirmation.
""")
    return decision
