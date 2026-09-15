from __future__ import annotations
import json
from pathlib import Path
from .util import require_new, write_json, write_csv, write_text, sha256_file

def write_final(art, out):
    art=Path(art); out=require_new(out)
    parent=json.loads((art/'00_static_v1'/'parent_validation.json').read_text()) if (art/'00_static_v1'/'parent_validation.json').is_file() else {}
    forensic=json.loads((art/'01_v6rc1_forensics_v1'/'summary.json').read_text()) if (art/'01_v6rc1_forensics_v1'/'summary.json').is_file() else {}
    sel=json.loads((art/'03_selection_v1'/'selected_model.json').read_text()) if (art/'03_selection_v1'/'selected_model.json').is_file() else {}
    hold=json.loads((art/'04_engineering_holdout_v1'/'decision.json').read_text()) if (art/'04_engineering_holdout_v1'/'decision.json').is_file() else {}
    status=hold.get('status') or sel.get('reason') or 'CLOSED_LOOP_REGRASP_ENGINEERING_NOT_READY'
    decision={'engineering_status':status,'confirmation_passed':False,'policy_gain_claimed':False,
              'visual_grounding_claimed':False,'real_robot_claimed':False,
              'selected_model':(sel.get('selected') or {}).get('model_id'),
              'parent':parent,'forensics':forensic,'holdout':hold,'V6_reward_changed':False,'RC1_changed':False}
    write_json(out/'decision.json', decision)
    write_csv(out/'claim_to_evidence.csv', [
        {'claim':'closed-loop regrasp ready','evidence':str(art/'04_engineering_holdout_v1'/'decision.json'),'status':status},
        {'claim':'confirmation_passed','evidence':'always false','status':False}])
    write_json(out/'next_stage_handoff.json', {'confirmation_data_reuse_allowed':False,'V6_CAP_POTENTIAL_frozen':True,
               'RC1_frozen':True,'selected_regrasp':decision['selected_model'],'confirmation_passed':False})
    write_json(out/'result_manifest.json', {'confirmation_passed':False,'status':status})
    write_text(out/'report.md', f'# V6RG1 closed-loop regrasp\n\nStatus: {status}\nconfirmation_passed: false\nSelected: {decision["selected_model"]}\nForensics: {json.dumps(forensic)}\nHoldout: {json.dumps(hold, default=str)[:2000]}\n')
    return decision
