#!/usr/bin/env python3
"""Draw the locked manual PathGraph method diagram; it has no result numbers."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--graph-spec-root',type=Path,required=True); p.add_argument('--reward-config',type=Path,required=True); p.add_argument('--output-base',type=Path,required=True); p.add_argument('--formats',required=True); p.add_argument('--dpi',type=int,default=300); p.add_argument('--source-map',type=Path,required=False); a=p.parse_args()
    c=yaml.safe_load(a.reward_config.read_text()); r=c.get('reward',c); fig,ax=plt.subplots(figsize=(10,5)); ax.axis('off')
    xy={'Start':(0.06,.52),'A':(.28,.76),'B':(.28,.28),'Goal':(.70,.52),'Failure':(.52,.10),'Recovery':(.52,.34)}
    for name,(x,y) in xy.items(): ax.scatter(x,y,s=1600,color='#e8f1fb',edgecolor='#1f4e79',zorder=2); ax.text(x,y,name,ha='center',va='center',fontsize=10,zorder=3)
    def arrow(left,right,label,color='#1f4e79',style='-'):
        x,y=xy[left]; u,v=xy[right]; ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=2,color=color,linestyle=style)); ax.text((x+u)/2,(y+v)/2+.04,label,ha='center',fontsize=9,color=color)
    arrow('Start','A','legal order A'); arrow('Start','B','legal order B'); arrow('A','Goal','alternative path'); arrow('B','Goal','alternative path'); arrow('Start','Failure','failure edge','#a61c00'); arrow('Failure','Recovery','recovery edge','#38761d'); arrow('Recovery','Goal','debt-capped credit','#38761d'); ax.annotate('',xy=(.94,.52),xytext=(.78,.52),arrowprops=dict(arrowstyle='->',lw=1.5,linestyle='--',color='#777')); ax.text(.86,.61,'RA-BC\nsecondary downstream use',ha='center',fontsize=9,color='#555')
    ax.text(.5,.94,'Locked manual PathGraph reward representation',ha='center',weight='bold',fontsize=13); ax.text(.5,.86,f"r = Δ remaining-cost + λ Δ within-node progress − η loop count\nlocked η={r.get('eta')}, β={r.get('beta')}; recovery credit is capped by failure debt",ha='center',fontsize=9)
    a.output_base.parent.mkdir(parents=True,exist_ok=True)
    for f in a.formats.split(','): fig.savefig(a.output_base.with_suffix('.'+f),dpi=a.dpi,bbox_inches='tight')
    if a.source_map:
        rows=[]
        if a.source_map.exists():
            with a.source_map.open(newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
        rows += [{'artifact_id':'figure_1_pathgraph_method','artifact_type':'figure','source_path':str(a.reward_config.resolve()),'source_kind':'locked_reward_config','statistics_unit_or_scope':'method schematic, no result values','unsupported_claim_in_main':False},{'artifact_id':'figure_1_pathgraph_method','artifact_type':'figure','source_path':str(a.graph_spec_root.resolve()),'source_kind':'locked_manual_graph_spec','statistics_unit_or_scope':'method schematic, no result values','unsupported_claim_in_main':False}]
        with a.source_map.open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
if __name__=='__main__': main()
