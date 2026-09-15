from __future__ import annotations
from .util import require_new, write_csv, write_json

def _p(mets, case):
    v=mets.get(case) or (0,6)
    return int(v[0]) if isinstance(v,(list,tuple)) else 0

def select(eval_root, contract, out):
    out=require_new(out)
    import json
    from pathlib import Path
    summ=json.loads((Path(eval_root)/'evaluation_summary.json').read_text(encoding='utf-8'))
    cands=summ.get('candidates') or []
    def ready_rg1(c):
        m=c.get('metrics') or {}
        return (c.get('hard_ok') and _p(m,'G1_LOSS_REGRASP_RETURN_P40')>=5 and _p(m,'G2_LOSS_REGRASP_RETURN_P80')>=5
                and _p(m,'G3_THREE_REGRASP_RETURN_P40')>=4 and _p(m,'G4_THREE_REGRASP_RETURN_P80')>=4
                and _p(m,'G5_HIGH_RESIDUAL_SPEED_P40')>=4 and _p(m,'G6_LOW_FRICTION_LONG_SLIDE_P80')>=4
                and _p(m,'G7_NO_LOSS_INITIAL_GRASP_CONTROL')>=6 and _p(m,'G8_COMMANDED_RELEASE_NO_REGRASP')>=6
                and (c.get('attempt1_rate_g16') or 0)>=0.80 and (c.get('cond_rc1_return') or 0)>=0.85
                and (c.get('p90_regrasp_s') or 9)<=3.0)
    def ready_rg2(c):
        m=c.get('metrics') or {}
        return (c.get('hard_ok') and _p(m,'G1_LOSS_REGRASP_RETURN_P40')>=5 and _p(m,'G2_LOSS_REGRASP_RETURN_P80')>=5
                and _p(m,'G3_THREE_REGRASP_RETURN_P40')>=4 and _p(m,'G4_THREE_REGRASP_RETURN_P80')>=4
                and _p(m,'G5_HIGH_RESIDUAL_SPEED_P40')>=4 and _p(m,'G6_LOW_FRICTION_LONG_SLIDE_P80')>=4
                and _p(m,'G7_NO_LOSS_INITIAL_GRASP_CONTROL')>=6 and _p(m,'G8_COMMANDED_RELEASE_NO_REGRASP')>=6
                and (c.get('all_attempt_rate_g16') or 0)>=0.90 and (c.get('mean_attempts') or 9)<=1.35
                and (c.get('cond_rc1_return') or 0)>=0.85 and (c.get('p90_regrasp_s') or 9)<=3.5)
    rg1=next((c for c in cands if c.get('model_id')=='RG1_TRACK_RECENTER_SINGLE_ATTEMPT'), None)
    rg2=next((c for c in cands if c.get('model_id')=='RG2_TRACK_RECENTER_ONE_RETRY'), None)
    selected=None; reason='NO_REGRASP_CANDIDATE_SELECTED'
    if rg1 and ready_rg1(rg1):
        selected=rg1; reason='select RG1 if it meets readiness'
    elif rg2 and ready_rg2(rg2):
        selected=rg2; reason='RG1 failed; RG2 meets readiness'
    write_csv(out/'candidate_comparison.csv', cands)
    write_json(out/'selection_trace.json', {'reason':reason,'rg1_ready':bool(rg1 and ready_rg1(rg1)),'rg2_ready':bool(rg2 and ready_rg2(rg2)),'reward_used':False})
    write_json(out/'selected_model.json', {'selected':selected,'reason':reason,'confirmation_passed':False})
    return selected
