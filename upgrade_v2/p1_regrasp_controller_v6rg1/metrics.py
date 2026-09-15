from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from .util import require_new, write_csv, write_json

G16=['G1_LOSS_REGRASP_RETURN_P40','G2_LOSS_REGRASP_RETURN_P80','G3_THREE_REGRASP_RETURN_P40','G4_THREE_REGRASP_RETURN_P80','G5_HIGH_RESIDUAL_SPEED_P40','G6_LOW_FRICTION_LONG_SLIDE_P80']

def evaluate_tree(data_root: Path, out: Path, holdout=False):
    out=require_new(out)
    data_root=Path(data_root)
    roots=[data_root] if (data_root/'collection_summary.json').is_file() else [p.parent for p in data_root.glob('*/collection_summary.json')]
    rows=[]
    for root in sorted(roots):
        for ident in root.rglob('identity.json'):
            dest=ident.parent
            rec=json.loads(ident.read_text(encoding='utf-8'))
            man=json.loads((dest/'manifest.json').read_text(encoding='utf-8')) if (dest/'manifest.json').is_file() else {}
            attempts=man.get('attempts') or []
            conf=any(a.get('status')=='REGRASP_CONFIRMED' for a in attempts if isinstance(a,dict))
            att1=any(a.get('status')=='REGRASP_CONFIRMED' and a.get('attempt')==1 for a in attempts if isinstance(a,dict))
            nret=sum(1 for x in (man.get('returns') or []) if isinstance(x,dict) and x.get('status')=='RETURN_COMPLETE')
            need=3 if rec.get('case_id') in ('G3_THREE_REGRASP_RETURN_P40','G4_THREE_REGRASP_RETURN_P80') else 1
            if rec.get('case_id')=='G8_COMMANDED_RELEASE_NO_REGRASP':
                case_pass = rec.get('regrasp_invocations',0)==0
            elif rec.get('case_id')=='G7_NO_LOSS_INITIAL_GRASP_CONTROL':
                case_pass = rec.get('regrasp_invocations',0)==0 and nret>=1
            else:
                case_pass = conf and nret>=need and man.get('direct_pose_overwrites_after_start',0)==0
            hard_ok = man.get('direct_pose_overwrites_after_start',0)==0 and man.get('checkpoint_resets_inside_episode',0)==0
            times=[a.get('time_s') for a in attempts if isinstance(a,dict) and a.get('time_s') is not None]
            rows.append(dict(rec, dest=str(dest), regrasp_confirmed=conf, attempt1=att1, nret=nret, need=need,
                             case_pass=case_pass, hard_ok=hard_ok, regrasp_time=min(times) if times else None,
                             n_attempts=len(attempts)))
    write_csv(out/'rollout_manifest.csv', rows)
    by=defaultdict(list)
    for r in rows: by[r.get('model_id') or 'UNK'].append(r)
    cand=[]; case_rows=[]
    for mid, rs in by.items():
        def fam_pass(case):
            fams=sorted({r['family_id'] for r in rs if r['case_id']==case})
            ok=sum(1 for f in fams if any(r['case_pass'] and r['family_id']==f and r['case_id']==case for r in rs))
            return ok,len(fams)
        mets={}
        for case in sorted({r['case_id'] for r in rs}):
            ok,n=fam_pass(case); mets[case]=(ok,n)
            case_rows.append({'model_id':mid,'case_id':case,'pass_families':ok,'n_families':n})
        times=sorted(r['regrasp_time'] for r in rs if r['regrasp_time'] is not None)
        g16=[r for r in rs if r['case_id'] in G16]
        att1_rate=sum(r['attempt1'] for r in g16)/max(1,len(g16))
        all_rate=sum(r['regrasp_confirmed'] for r in g16)/max(1,len(g16))
        cond_ret=sum(r['nret']>0 for r in g16 if r['regrasp_confirmed'])/max(1,sum(r['regrasp_confirmed'] for r in g16))
        mean_att=sum(r['n_attempts'] for r in g16)/max(1,len(g16))
        p90=times[int(0.9*(len(times)-1))] if times else None
        med=times[len(times)//2] if times else None
        cand.append({'model_id':mid,'hard_ok':all(r['hard_ok'] for r in rs),'metrics':mets,
                     'attempt1_rate_g16':att1_rate,'all_attempt_rate_g16':all_rate,
                     'cond_rc1_return':cond_ret,'mean_attempts':mean_att,
                     'median_regrasp_s':med,'p90_regrasp_s':p90,'n':len(rs),
                     'failures':sum(not r['case_pass'] for r in rs)})
    write_csv(out/('holdout_metrics.csv' if holdout else 'candidate_metrics.csv'), cand)
    write_csv(out/'per_case_family_results.csv', case_rows)
    write_json(out/'evaluation_summary.json', {'n':len(rows),'candidates':cand})
    return {'rollouts':rows,'candidates':cand}
