import json,time
import numpy as np
from common import ROOT,write

def main():
    rows=[];references={};hash_match=True;total_calls=total_steps=total_updates=0
    for seed in (0,3):
        for i in range(6):
            arm=f'R{i}';d=ROOT/'runs'/f's{seed}_{arm}'
            row=dict(seed=seed,arm=arm,status='pending')
            if (d/'probes.json').exists():
                p=json.loads((d/'probes.json').read_text());b=p[0];flags=[]
                for x in p[1:]:flags.append((x['V_online_start']-x['V_online_all'])-(b['V_online_start']-b['V_online_all'])>=15 and x['dqda_norm_start']>=2*b['dqda_norm_start'])
                longest=cur=0
                for f in flags:cur=cur+1 if f else 0;longest=max(longest,cur)
                row.update(updates=p[-1]['updates'],anomaly_points=sum(flags),longest_anomaly_run=longest,
                           value_gap_final=p[-1]['V_online_start']-p[-1]['V_online_all'],status='running')
                total_updates+=p[-1]['updates']
            if (d/'complete.json').exists():row.update(status='completed',cost=json.loads((d/'complete.json').read_text()))
            for it in (50000,100000):
                path=d/f'eval_{it}/results.json'
                if not path.exists():continue
                e=json.loads(path.read_text());total_calls+=e['control_calls'];total_steps+=e['external_steps']
                hashes={x['episode']:x['initial_obs_sha256'] for x in e['episodes']}
                if references:hash_match &= hashes==references
                else:references=hashes
                row[f'eval_{it}']={k:e[k] for k in ('success_percent','start_stay_percent','return_mean','control_calls','seconds')}
            rows.append(row)
    assert hash_match,'matched initial states differ'
    assert total_calls<=5000000 and total_updates<=1200000
    write(ROOT/'reports/summary.json',dict(time=time.time(),rows=rows,matched_initial_hashes=hash_match,
          total_recorded_updates=total_updates,total_eval_control_calls=total_calls,total_external_eval_steps=total_steps))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
