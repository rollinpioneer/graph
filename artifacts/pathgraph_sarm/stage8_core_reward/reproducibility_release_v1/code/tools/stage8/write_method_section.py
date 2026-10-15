#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
def main():
 p=argparse.ArgumentParser();p.add_argument('--graph-spec-root',type=Path,required=True);p.add_argument('--reward-config',type=Path,required=True);p.add_argument('--model-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--equations',type=Path,required=True);a=p.parse_args();cfg=yaml.safe_load(a.reward_config.read_text());r=cfg.get('reward',cfg); specs=sorted(x.name for x in a.graph_spec_root.glob('*.yaml'))
 eq='\\[\nr_t = C_G(z_t,h_t)-C_G(z_{t+1},h_{t+1}) + \\lambda[\\phi_{z_t}(o_{t+1})-\\phi_{z_t}(o_t)] - \\eta n_{\\mathrm{loop}}(e_t)\n\\]\n\\[\nw_t=\\max(0,\\mathbb E[r_t]-\\beta\\operatorname{Std}[r_t])\n\\]\n'
 a.equations.parent.mkdir(parents=True,exist_ok=True);a.equations.write_text(eq,encoding='utf-8');text=f'''# Method\n\nWe use the frozen manual runtime graph specifications `{', '.join(specs)}`. A checkpoint ensemble predicts graph-node and edge distributions, remaining cost, and within-node progress. The reward engine then applies the fixed graph transition rule in `method_equations.tex`. The locked values are lambda={r.get('lambda')}, eta={r.get('eta')}, and beta={r.get('beta')}. Because eta and beta are zero, nonzero loop penalties and uncertainty LCB are not presented as validated main-result contributions. Recovery credit is bounded by accumulated failure debt.\n''';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
