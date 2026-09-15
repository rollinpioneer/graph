from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
from .util import require_new, write_csv, write_json
from .tracking_controller import xy_error, z_error, norm

def _jsonl(p):
    p=Path(p)
    if not p.is_file(): return []
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

def audit_episode(dest: Path):
    ident=json.loads((dest/'identity.json').read_text(encoding='utf-8'))
    notes=ident.get('notes') or []
    man=json.loads((dest/'manifest.json').read_text(encoding='utf-8')) if (dest/'manifest.json').is_file() else {}
    raw=_jsonl(dest/'raw_state.jsonl')
    failed='regrasp_failed' in str(notes)
    tax='R8_LOG_OR_BOUNDARY_UNRESOLVED'
    close_xy=close_z=close_speed=None
    if not failed:
        tax='OK_OR_NOT_TARGET'
    elif raw:
        # find CLOSE mode frames
        closes=[r for r in raw if r.get('controller_mode') in ('CLOSE','DESCEND','SETTLE')]
        if closes:
            r=closes[-1]; o=r['objects']['obj']
            close_xy=xy_error(r['eef'], o['pos']); close_z=z_error(r['eef'], o['pos'])
            close_speed=norm(o['vel'])
            rel=math.hypot(close_xy, close_z)
            if rel>0.02: tax='R5_CLOSE_OUTSIDE_CAPTURE_RANGE'
            elif close_speed>0.03: tax='R2_OBJECT_NOT_SETTLED'
            else: tax='R6_WELD_TRIGGERED_HOLD_NOT_STABLE'
        else:
            tax='R3_APPROACH_TARGET_NOT_REACHED'
    return {'episode_id':ident.get('episode_id'),'case_id':ident.get('case_id'),'family_id':ident.get('family_id'),
            'notes':notes,'taxonomy':tax,'close_xy':close_xy,'close_z':close_z,'close_speed':close_speed,
            'failed':failed,'raw':str(dest)}

def run_forensics(raw_root: Path, out: Path):
    out=require_new(out)
    rows=[]
    for ident in Path(raw_root).rglob('identity.json'):
        rec=json.loads(ident.read_text(encoding='utf-8'))
        if 'regrasp_failed' not in str(rec.get('notes')):
            continue
        rows.append(audit_episode(ident.parent))
    write_csv(out/'attempt_forensics.csv', rows)
    tax=defaultdict(int)
    for r in rows: tax[r['taxonomy']]+=1
    write_csv(out/'failure_taxonomy.csv', [{'taxonomy':k,'n':v} for k,v in sorted(tax.items())])
    write_csv(out/'close_time_alignment.csv', rows)
    write_csv(out/'object_motion_during_grasp.csv', rows)
    summary={'n_failed_regrasp':len(rows),'taxonomy':dict(tax)}
    write_json(out/'summary.json', summary)
    return summary
