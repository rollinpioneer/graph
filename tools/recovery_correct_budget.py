#!/usr/bin/env python3
import csv,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/v13_r3_recovery'; REPORT=ROOT/'reports/v13_r3_recovery'
def write(p,d): pathlib.Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 p=OUT/'recovery_authorization.json'; d=json.loads(p.read_text()); d['new_eval_episode_budget_total']=38; d['new_eval_episode_budget_b2']=20; d['full_prior_switch_new_episode_budget']=18; d.pop('new_eval_episode_budget',None); write(p,d)
 p=ROOT/'status/v13_r3_recovery.json'; d=json.loads(p.read_text()); d['new_eval_episodes']=38; write(p,d)
 p=REPORT/'cost_ledger.csv'; rows=list(csv.reader(p.open())); rows=[['new_eval_episodes','38','20 B2 dev10 + 18 Full prior switch'] if r and r[0]=='new_eval_episodes' else r for r in rows];
 with p.open('w',newline='') as f: csv.writer(f).writerows(rows)
 p=REPORT/'Results_R3_Recovery.md'; s=p.read_text(); s=s.replace('The new Full@Absent case succeeded with return 0.6271314720175796.','Full@Absent covered all 10 dev10 cases (10/10 success; mean start discounted return 0.6350014580965666). The frozen nonempty-prior stratum remains D0_dev_00.'); p.write_text(s)
 p=REPORT/'matched_comparison.csv'; rows=list(csv.DictReader(p.open()));
 for r in rows:
  if r['comparison']=='B2@Empty': r['episodes']='10'
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
