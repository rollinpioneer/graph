#!/usr/bin/env python3
import argparse,json,yaml
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--graph',required=True); ap.add_argument('--schema'); ap.add_argument('--report'); a=ap.parse_args(); g=yaml.safe_load(open(a.graph)); ns={n['id'] for n in g['nodes']}; assert len(ns)==len(g['nodes']); assert g['start_node'] in ns and set(g['success_nodes'])<=ns
    for e in g['edges']: assert e['src'] in ns and e['dst'] in ns and e['type'] in {'forward','alternative','failure','recovery','stagnation'}
    assert g['path_templates'] and all(p[0]==g['start_node'] and p[-1] in g['success_nodes'] for p in g['path_templates'])
    out={'valid':True,'nodes':len(ns),'edges':len(g['edges']),'paths':len(g['path_templates'])}; print(json.dumps(out))
    if a.report: open(a.report,'w').write(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
