#!/usr/bin/env python3
"""Group-level bootstrap helper for the Stage 3 content-group metrics table."""
import argparse, csv
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', type=Path, required=True); ap.add_argument('--group-column', default='content_group_id')
    ap.add_argument('--resamples', type=int, default=2000); ap.add_argument('--seed', type=int, default=20260903); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(a.metrics.open(newline='')))
    if not rows: raise SystemExit('empty metrics table')
    group_key = a.group_column
    groups = {}
    for row in rows: groups.setdefault(row[group_key], []).append(row)
    representatives = [v[0] for v in groups.values()]
    exclude = {group_key,'episode_id','scenario','case_type','outcome','method','task_id','orientation','seed','n_episode_rows'}
    out=[]; rng=np.random.default_rng(a.seed)
    for field in rows[0]:
        if field in exclude: continue
        vals=[]
        for row in representatives:
            try: vals.append(float(row[field]))
            except (KeyError, TypeError, ValueError): pass
        if not vals: continue
        vals=np.asarray(vals); boot=np.asarray([rng.choice(vals,len(vals),replace=True).mean() for _ in range(a.resamples)])
        out.append({'metric':field,'method':'mixed','task_id':'mixed','orientation':'mixed','seed_group':'mixed','mean':float(vals.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'bootstrap_unit':group_key,'n_units':len(vals),'n_episode_rows':len(rows)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
if __name__=='__main__': main()
