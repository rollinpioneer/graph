#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
def mk(task,tid,nodes,edges,types,costs,phis=None,attempt=0,terminal=None,prop=''):
 phis=phis or [0.0]*len(nodes); terminal=terminal or [False]*len(nodes); rows=[]
 for i,(n,e,t,c,ph) in enumerate(zip(nodes,edges,types,costs,phis)):
  rows.append({'task_id':task,'trace_id':tid,'step':i,'node_id':n,'edge_id':e,'edge_type':t,'phi':ph,'remaining_cost':c,'attempt_id':attempt,'is_terminal':terminal[i],'expected_property':prop})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--graph-spec-root');p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--manifest',required=True);a=p.parse_args(); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); allrows=[]
 # Dual-order graph: start=2, A=0, B=1, success=3.
 if 'transport_dual_order' in a.tasks:
  specs=[('legal_A_then_B',[2,0,1,3],[4,0,2,3],[1,1,1,1],[3,2,1,0]),('legal_B_then_A',[2,1,0,3],[5,2,0,1],[1,1,1,1],[3,2,1,0]),('legal_shortest_success',[2,0,3],[4,1,1],[1,1,1],[2,1,0])]
  for tid,n,e,t,c in specs: allrows += mk('transport_dual_order',tid,n,e,t,c,prop='legal_path')
 # Recovery graph: start=5, grasped=1, transit=2, placed=3, recovery=4, success=6.
 if 'transport_recovery' in a.tasks:
  specs=[('failure_then_recovery',[5,1,2,7,4,1,2,3,6],[7,1,4,0,6,1,3,3,5],[1,1,4,3,3,1,1,1,1],[5,4,3,4,3,2,1,0,0]),('failure_recovery_loop_x1',[5,1,2,7,4,1,2,7,4,1,2,3,6],[7,1,4,0,6,1,3,0,6,1,3,3,5],[1,1,4,3,3,1,1,4,3,1,1,1,1],[7,6,5,6,5,4,3,4,3,2,1,0,0]),('failure_recovery_loop_x2',[5,1,2,7,4,1,2,7,4,1,2,7,4,1,2,3,6],[7,1,4,0,6,1,3,0,6,1,3,0,6,1,3,3,5],[1,1,4,3,3,1,1,4,3,1,1,4,3,1,1,1,1],[9,8,7,8,7,6,5,6,5,4,3,4,3,2,1,0,0]),('failure_recovery_loop_x3',[5,1,2,7,4,1,2,7,4,1,2,7,4,1,2,7,4,1,2,3,6],[7,1,4,0,6,1,3,0,6,1,3,0,6,1,3,0,6,1,3,3,5],[1,1,4,3,3,1,1,4,3,1,1,4,3,1,1,4,3,1,1,1,1],[11,10,9,10,9,8,7,8,7,6,5,6,5,4,3,4,3,2,1,0,0]),('stagnation_same_node',[5,5,5,5],[0,0,0,0],[5,5,5,5],[2,2,2,2]),('illegal_backtrack',[5,1,5,1,2,3,6],[7,1,7,1,3,3,5],[1,1,2,2,1,1,1],[5,4,4,3,2,1,0]),('terminal_success_hold',[5,1,2,3,6,6,6],[7,1,3,5,5,5,5],[1,1,1,1,0,0,0],[4,3,2,1,0,0,0])]
  for tid,n,e,t,c in specs: allrows += mk('transport_recovery',tid,n,e,t,c,terminal=[(tid=='terminal_success_hold' and i>=4) for i in range(len(n))] if tid=='terminal_success_hold' else None,prop=tid)
 by={}
 for r in allrows: by.setdefault(r['trace_id'],[]).append(r)
 m=Path(a.manifest);m.parent.mkdir(parents=True,exist_ok=True)
 with m.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['trace_id','task_id','path','rows','expected_property','cost_convention']);w.writeheader()
  for tid,rs in by.items():
   q=out/(tid+'.jsonl');q.write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in rs)+'\n');w.writerow({'trace_id':tid,'task_id':rs[0]['task_id'],'path':str(q.resolve()),'rows':len(rs),'expected_property':rs[0]['expected_property'],'cost_convention':'unit legal; failure +1; recovery -1 debt-capped'})
if __name__=='__main__': main()
