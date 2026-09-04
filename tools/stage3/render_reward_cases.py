#!/usr/bin/env python3
"""Render fixed, pre-frozen diagnostic cases without selecting on effect size."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--suite-dir',type=Path,required=True); ap.add_argument('--metrics-dir',type=Path,required=True); ap.add_argument('--prediction-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--case-index',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    diag=list(csv.DictReader((a.suite_dir/'tables/diagnostic_episodes.csv').open())); by={x['episode_id']:x for x in diag}
    preds=[]
    for p in sorted(a.prediction_root.rglob('*.jsonl')):
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if rows: preds.append((p,rows))
    def choose(case,scenario=None):
        for d in diag:
            if d['case_type']==case and (scenario is None or d['scenario']==scenario): return d
        return None
    cases=[choose('canonical_chain','order_A_then_B'),choose('alternative_order'),choose('recovery_success'),choose('terminal_failure')]
    cases=[x for x in cases if x]
    lines=['# Fixed diagnostic cases','', 'Cases were selected from the frozen suite by case type, never by score magnitude.','']
    for d in cases:
        lines.append(f"- `{d['episode_id']}`: {d['case_type']} ({d['scenario']}), content_group_id={d['content_group_id']}")
        for p,rows in preds:
            sub=[r for r in rows if r.get('episode_id')==d['episode_id']]
            if not sub: continue
            sub.sort(key=lambda r:int(r.get('step',0))); x=np.asarray([float(r.get('progress',0)) for r in sub]); y=np.asarray([float(r.get('reward_delta',0)) for r in sub])
            fig,ax=plt.subplots(figsize=(7,3)); ax.plot(x,label='progress'); ax.plot(y,label='reward_delta',alpha=.7); ax.set_title(f"{d['case_type']} | {sub[0].get('method')} | {sub[0].get('orientation')} | seed={sub[0].get('seed')}"); ax.legend(); fig.tight_layout(); fig.savefig(a.output_dir/f"{d['case_type']}_{sub[0].get('method')}_{sub[0].get('orientation')}_{sub[0].get('seed')}.png"); plt.close(fig)
            break
    a.case_index.parent.mkdir(parents=True,exist_ok=True); a.case_index.write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
